#!/usr/bin/env python3
"""Compare 32x32 fp8 kernel output against torch reference computation.

This test:
1. Creates simple MoE weights, quantizes to MXFP4
2. Runs the fused_moe kernel (both 16x16 and 32x32)
3. Computes a torch reference: dequantize fp4 → float matmul
4. Compares kernel gemm1 output against reference
"""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, cktile_moe_stage1, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

EXPERTS = 2
TOPK = 1  # Simplify routing
DTYPE = dtypes.bf16

def round_up(x, a):
    return ((x + a - 1) // a) * a


def dequant_mxfp4(w_qt_raw, w_scale_raw, E, N, K):
    """Dequantize MXFP4 weights to float32 for reference computation.

    w_qt_raw: (E, N, K//2) uint8 — BEFORE shuffling
    w_scale_raw: (E*N, K//32) e8m0 — BEFORE shuffling
    """
    # Convert fp4x2 packed bytes to float
    # Each byte has 2 fp4 values: high nibble and low nibble
    w_bytes = w_qt_raw.view(torch.uint8)  # (E, N, K//2)

    # Unpack: low nibble first, then high nibble
    low = (w_bytes & 0x0F).to(torch.float32)
    high = ((w_bytes >> 4) & 0x0F).to(torch.float32)

    # fp4 E2M1 decoding:
    # bit3=sign, bit2-1=exponent, bit0=mantissa
    def fp4_to_float(x):
        sign = ((x.to(torch.int32) >> 3) & 1).float()
        exp = ((x.to(torch.int32) >> 1) & 3).float()
        mant = (x.to(torch.int32) & 1).float()

        # E2M1: value = (-1)^sign * 2^(exp-1) * (1 + mant/2) for exp > 0
        #        value = (-1)^sign * mant/2                    for exp == 0
        val = torch.where(
            exp > 0,
            (1.0 - 2.0 * sign) * (2.0 ** (exp - 1)) * (1.0 + mant * 0.5),
            (1.0 - 2.0 * sign) * mant * 0.5
        )
        return val

    low_f = fp4_to_float(low)   # (E, N, K//2)
    high_f = fp4_to_float(high)  # (E, N, K//2)

    # Interleave: element 0 = low[0], element 1 = high[0], element 2 = low[1], ...
    w_float = torch.zeros((E, N, K), dtype=torch.float32, device=w_qt_raw.device)
    w_float[:, :, 0::2] = low_f
    w_float[:, :, 1::2] = high_f

    # Apply MX block scale (per 32-element group along K)
    # w_scale_raw: (E*N, K//32) e8m0
    scale_uint8 = w_scale_raw.view(torch.uint8)  # (E*N, K//32)
    scale_float = (2.0 ** (scale_uint8.float() - 127.0))  # Convert E8M0 to float

    # Reshape scale to broadcast: (E, N, K//32, 1) → (E, N, K)
    scale_float = scale_float.view(E, N, K // 32, 1).expand(E, N, K // 32, 32).reshape(E, N, K)

    return w_float * scale_float


def test_reference():
    """Compare kernel output against torch reference."""
    hidden_raw = 128
    inter_raw = 128
    M = 512  # triggers fp8 path

    for n_lane, k_align, label, warp32 in [
        (16, 256, "16x16", "0"),
        (32, 128, "32x32", "1"),
    ]:
        os.environ["AITER_MOE_WARP32"] = warp32

        hidden = round_up(hidden_raw, k_align)
        inter = round_up(inter_raw, k_align)
        hidden_pad = hidden - hidden_raw
        inter_pad = inter - inter_raw

        torch.manual_seed(42)

        # Create raw weights at padded dimensions
        w1_raw = torch.randn((EXPERTS, inter * 2, hidden), dtype=DTYPE) / 10
        w2_raw = torch.randn((EXPERTS, hidden, inter), dtype=DTYPE) / 10

        # Quantize
        torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
        w1_qt, w1_scale = torch_quant(w1_raw, quant_dtype=dtypes.fp4x2)
        w1_qt = w1_qt.view(w1_raw.shape[0], w1_raw.shape[1], w1_raw.shape[2] // 2)

        # Keep UN-SHUFFLED copies for reference
        w1_qt_raw = w1_qt.clone()
        w1_scale_raw = w1_scale.clone()

        # Compute torch reference: dequant → matmul
        w1_deq = dequant_mxfp4(w1_qt_raw, w1_scale_raw, EXPERTS, inter * 2, hidden)

        print(f"\n{'='*80}")
        print(f"{label}: hidden={hidden} (+{hidden_pad}), inter={inter} (+{inter_pad})")
        print(f"{'='*80}")
        print(f"  w1_deq stats: mean={w1_deq.abs().mean():.6f} max={w1_deq.abs().max():.6f}")

        # Create input
        torch.manual_seed(200)
        hidden_states_raw = torch.randn((M, hidden_raw), dtype=DTYPE)

        # Pad input
        if hidden_pad > 0:
            hidden_states = torch.nn.functional.pad(hidden_states_raw, (0, hidden_pad), value=0.0)
        else:
            hidden_states = hidden_states_raw.clone()

        # Torch reference gemm1 (no MoE routing, just straight matmul with expert 0)
        # gemm1 output before activation = hidden_states @ w1[expert].T
        hs_float = hidden_states.float()
        ref_gemm1 = hs_float @ w1_deq[0].T  # (M, inter*2)

        # Apply swiglu
        gate = ref_gemm1[:, :inter]
        up = ref_gemm1[:, inter:]
        ref_swiglu = torch.nn.functional.silu(gate) * up

        print(f"  Torch reference gemm1: mean={ref_gemm1.abs().mean():.6f} max={ref_gemm1.abs().max():.6f}")
        print(f"  Torch reference swiglu: mean={ref_swiglu.abs().mean():.6f} max={ref_swiglu.abs().max():.6f}")

        # Now run the actual kernel
        # Shuffle weights
        w1_qt_s = shuffle_weight_a16w4(w1_qt, n_lane, True)
        w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
        w1_qt_s.is_shuffled = True

        # Convert to fp8
        a1 = hidden_states.to(dtypes.fp8)

        # Create trivial MoE routing: all tokens → expert 0
        score = torch.zeros((M, EXPERTS), dtype=DTYPE)
        score[:, 0] = 1.0  # all tokens go to expert 0
        topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

        # MoE sorting
        block_m = 64 if warp32 == "1" else 32
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
            topk_ids, topk_weights, EXPERTS, inter * 2,  # model_dim = output dim of gemm1
            moebuf_dtype=DTYPE, block_size=block_m,
        )

        # Create a1_scale (ones)
        a1_scale = torch.ones([sorted_ids.shape[0], a1.shape[-1] // 32],
                              dtype=dtypes.fp8_e8m0, device=a1.device)

        # View as fp4x2
        w1_v = w1_qt_s
        if hasattr(torch, "float4_e2m1fn_x2"):
            w1_v = w1_qt_s.view(torch.float4_e2m1fn_x2)
            w1_v.is_shuffled = True

        # Dummy w2 for stage1 (needed for shape computation)
        w2_dummy = torch.zeros((EXPERTS, hidden, inter // 2), dtype=torch.uint8, device=w1_qt_s.device)
        if hasattr(torch, "float4_e2m1fn_x2"):
            w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

        torch.cuda.synchronize()
        try:
            out = cktile_moe_stage1(
                a1,
                w1_v,
                w2_dummy,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                None,  # out
                TOPK,
                block_m,
                a1_scale=a1_scale,
                w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
                activation=ActivationType.Swiglu,
            )
            torch.cuda.synchronize()

            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            out_mean = out.float().abs().mean().item()
            out_max = out.float().abs().max().item() if not has_nan else float('nan')

            print(f"  Kernel gemm1+swiglu: mean={out_mean:.6f} max={out_max:.6f} nan={has_nan} inf={has_inf}")
            print(f"  Shape: kernel={out.shape} ref={ref_swiglu.shape}")

            if not has_nan:
                # Compare first few tokens (expert 0)
                # The kernel output has shape (M, topk, inter_dim)
                kernel_out = out[:, 0, :inter].float()  # (M, inter)
                ref_out = ref_swiglu[:, :inter].float()

                ratio = kernel_out.abs().mean() / max(ref_out.abs().mean().item(), 1e-10)
                cos_sim = torch.nn.functional.cosine_similarity(
                    kernel_out.reshape(1, -1), ref_out.reshape(1, -1)
                ).item()
                print(f"  Ratio (kernel/ref): {ratio:.4f}")
                print(f"  Cosine similarity: {cos_sim:.6f}")
            else:
                print(f"  Cannot compare: kernel output has NaN")
                # Print first non-NaN values
                flat = out.float().reshape(-1)
                non_nan = flat[~torch.isnan(flat)]
                if len(non_nan) > 0:
                    print(f"  Non-NaN values: {len(non_nan)}/{len(flat)}, mean={non_nan.abs().mean():.6f}")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    os.environ["AITER_DEBUG_MOE"] = "0"
    test_reference()
