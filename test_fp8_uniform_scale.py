#!/usr/bin/env python3
"""Test fp8 32x32 with UNIFORM scales to isolate scale loading from data path.

If scales are uniform (all bytes identical), opsel and scale loading issues
cannot cause magnitude errors. Any error must be in the data path (A/B loading).
"""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, fused_moe
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

EXPERTS = 4
TOPK = 2
DTYPE = dtypes.bf16
ACTIVATION = ActivationType.Swiglu

def round_up(x, a):
    return ((x + a - 1) // a) * a


def test_uniform_scale():
    """Test with all scales = 2^(-4) = E8M0 value 123."""
    # Use a moderate scale so output doesn't overflow fp8
    SCALE_BYTE = 123  # 2^(123-127) = 2^(-4) = 0.0625

    for n_lane, k_align, label, warp32 in [
        (16, 256, "16x16", "0"),
        (32, 128, "32x32", "1"),
    ]:
        os.environ["AITER_MOE_WARP32"] = warp32

        hidden_raw = 128
        inter_raw = 128
        hidden = round_up(hidden_raw, k_align)
        inter = round_up(inter_raw, k_align)
        hidden_pad = hidden - hidden_raw
        inter_pad = inter - inter_raw

        torch.manual_seed(42)

        # Create weights
        w1_raw = torch.randn((EXPERTS, inter * 2, hidden), dtype=DTYPE) / 10
        w2_raw = torch.randn((EXPERTS, hidden, inter), dtype=DTYPE) / 10

        # Quantize
        torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
        w1_qt, w1_scale = torch_quant(w1_raw, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = torch_quant(w2_raw, quant_dtype=dtypes.fp4x2)
        w1_qt = w1_qt.view(w1_raw.shape[0], w1_raw.shape[1], w1_raw.shape[2] // 2)
        w2_qt = w2_qt.view(w2_raw.shape[0], w2_raw.shape[1], w2_raw.shape[2] // 2)

        # Force ALL scales to uniform value BEFORE shuffling
        w1_scale_uniform = torch.full_like(w1_scale.view(torch.uint8), SCALE_BYTE).view(w1_scale.dtype)
        w2_scale_uniform = torch.full_like(w2_scale.view(torch.uint8), SCALE_BYTE).view(w2_scale.dtype)

        # Shuffle weights
        w1_qt = shuffle_weight_a16w4(w1_qt, n_lane, True)
        w1_scale_s = shuffle_scale_a16w4(w1_scale_uniform, EXPERTS, True, n_lane=n_lane)
        w2_qt = shuffle_weight_a16w4(w2_qt, n_lane, False)
        w2_scale_s = shuffle_scale_a16w4(w2_scale_uniform, EXPERTS, False, n_lane=n_lane)
        w1_qt.is_shuffled = True
        w2_qt.is_shuffled = True

        print(f"\n{'='*80}")
        print(f"{label}: hidden={hidden} (+{hidden_pad}), inter={inter} (+{inter_pad})")
        print(f"  All scales = 2^({SCALE_BYTE}-127) = 2^({SCALE_BYTE-127})")
        print(f"{'='*80}")

        # Verify scales are uniform after shuffling
        s_bytes = w1_scale_s.view(torch.uint8)
        print(f"  w1_scale after shuffle: min={s_bytes.min().item()} max={s_bytes.max().item()} unique={s_bytes.unique().numel()}")

        # View as fp4x2
        w1_v = w1_qt
        w2_v = w2_qt
        if hasattr(torch, "float4_e2m1fn_x2"):
            w1_v = w1_qt.view(torch.float4_e2m1fn_x2)
            w2_v = w2_qt.view(torch.float4_e2m1fn_x2)
            w1_v.is_shuffled = True
            w2_v.is_shuffled = True

        # Test bf16 path (M=256)
        torch.manual_seed(200)
        M = 256
        hidden_states = torch.randn((M, hidden_raw), dtype=DTYPE)
        score = torch.randn((M, EXPERTS), dtype=DTYPE)
        topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
        if hidden_pad > 0:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, hidden_pad), value=0.0)

        out_bf16 = fused_moe(
            hidden_states=hidden_states, w1=w1_v, w2=w2_v,
            w1_scale=w1_scale_s, w2_scale=w2_scale_s, a1_scale=None,
            topk_weight=topk_weights, topk_ids=topk_ids,
            quant_type=QuantType.per_1x32, activation=ACTIVATION,
            expert_mask=None, num_local_tokens=None, dtype=DTYPE,
            hidden_pad=hidden_pad, intermediate_pad=inter_pad,
        )
        if hidden_pad > 0:
            out_bf16 = out_bf16[..., :hidden_raw]
        has_nan = torch.isnan(out_bf16).any().item()
        print(f"  bf16 (M=256): mean={out_bf16.abs().mean().item():.6f} max={out_bf16.abs().max().item():.6f} nan={has_nan}")

        # Test fp8 path (M=512)
        torch.manual_seed(200)
        M = 512
        hidden_states = torch.randn((M, hidden_raw), dtype=DTYPE)
        score = torch.randn((M, EXPERTS), dtype=DTYPE)
        topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
        if hidden_pad > 0:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, hidden_pad), value=0.0)

        out_fp8 = fused_moe(
            hidden_states=hidden_states, w1=w1_v, w2=w2_v,
            w1_scale=w1_scale_s, w2_scale=w2_scale_s, a1_scale=None,
            topk_weight=topk_weights, topk_ids=topk_ids,
            quant_type=QuantType.per_1x32, activation=ACTIVATION,
            expert_mask=None, num_local_tokens=None, dtype=DTYPE,
            hidden_pad=hidden_pad, intermediate_pad=inter_pad,
        )
        if hidden_pad > 0:
            out_fp8 = out_fp8[..., :hidden_raw]
        has_nan = torch.isnan(out_fp8).any().item()
        has_inf = torch.isinf(out_fp8).any().item()
        non_nan = out_fp8[~torch.isnan(out_fp8)]
        non_nan_mean = non_nan.abs().mean().item() if len(non_nan) > 0 else float('nan')
        print(f"  fp8  (M=512): mean={non_nan_mean:.6f} max={non_nan.abs().max().item() if len(non_nan) > 0 else float('nan'):.6f} nan={has_nan} inf={has_inf}")
        if has_nan:
            nan_count = torch.isnan(out_fp8).sum().item()
            total = out_fp8.numel()
            print(f"    NaN: {nan_count}/{total} ({100*nan_count/total:.1f}%)")

        # Compare
        if not torch.isnan(out_bf16).any() and not torch.isnan(out_fp8).any():
            # Can't directly compare since M differs, but can compare magnitudes
            ratio = out_fp8.abs().mean().item() / max(out_bf16.abs().mean().item(), 1e-10)
            print(f"  Magnitude ratio (fp8/bf16): {ratio:.2f}x")


if __name__ == "__main__":
    os.environ["AITER_DEBUG_MOE"] = "0"
    test_uniform_scale()
