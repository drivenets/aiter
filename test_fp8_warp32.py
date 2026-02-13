#!/usr/bin/env python3
"""Minimal test for 32x32 fp8+fp4 MoE kernel.

Tests:
1. 16x16 fp8 baseline (known working)
2. 32x32 fp8 kernel
3. Varies K dimension to test odd/even K-block counts
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

EXPERTS = 4  # small for fast test
TOPK = 2
DTYPE = dtypes.bf16
ACTIVATION = ActivationType.Swiglu

def round_up(x, a):
    return ((x + a - 1) // a) * a


def test_kernel(label, hidden_raw, inter_raw, k_align, n_lane, M, seed=42):
    """Test a single kernel configuration."""
    hidden = round_up(hidden_raw, k_align)
    inter = round_up(inter_raw, k_align)
    hidden_pad = hidden - hidden_raw
    inter_pad = inter - inter_raw

    k_blocks_gemm1 = hidden // (256 if n_lane == 16 else 128)
    k_blocks_gemm2 = inter // (256 if n_lane == 16 else 128)

    torch.manual_seed(seed)

    # Create and quantize weights
    w1_raw = torch.randn((EXPERTS, inter * 2, hidden), dtype=DTYPE) / 10
    w2_raw = torch.randn((EXPERTS, hidden, inter), dtype=DTYPE) / 10

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1_raw, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2_raw, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(w1_raw.shape[0], w1_raw.shape[1], w1_raw.shape[2] // 2)
    w2_qt = w2_qt.view(w2_raw.shape[0], w2_raw.shape[1], w2_raw.shape[2] // 2)

    # Shuffle
    w1_qt = shuffle_weight_a16w4(w1_qt, n_lane, True)
    w1_scale = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
    w2_qt = shuffle_weight_a16w4(w2_qt, n_lane, False)
    w2_scale = shuffle_scale_a16w4(w2_scale, EXPERTS, False, n_lane=n_lane)
    w1_qt.is_shuffled = True
    w2_qt.is_shuffled = True

    # Create input
    torch.manual_seed(seed + 100)
    hidden_states = torch.randn((M, hidden_raw), dtype=DTYPE)
    score = torch.randn((M, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    # Pad input
    if hidden_pad > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, hidden_pad), value=0.0)

    # View as fp4x2
    w1_v = w1_qt
    w2_v = w2_qt
    if hasattr(torch, "float4_e2m1fn_x2"):
        w1_v = w1_qt.view(torch.float4_e2m1fn_x2)
        w2_v = w2_qt.view(torch.float4_e2m1fn_x2)
        w1_v.is_shuffled = True
        w2_v.is_shuffled = True

    try:
        out = fused_moe(
            hidden_states=hidden_states,
            w1=w1_v,
            w2=w2_v,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            quant_type=QuantType.per_1x32,
            activation=ACTIVATION,
            expert_mask=None,
            num_local_tokens=None,
            dtype=DTYPE,
            hidden_pad=hidden_pad,
            intermediate_pad=inter_pad,
        )

        if hidden_pad > 0:
            out = out[..., :hidden_raw]

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        all_zero = (out == 0).all().item()
        out_mean = out.abs().mean().item()

        status = "FAIL" if (has_nan or has_inf or all_zero or out_mean > 1000) else "PASS"
        details = f"mean={out_mean:.6f} nan={has_nan} inf={has_inf} zero={all_zero}"

        print(f"  {label} M={M:>5} | K_gemm1={hidden:>5} ({k_blocks_gemm1:>3} blocks) | "
              f"K_gemm2={inter:>5} ({k_blocks_gemm2:>3} blocks) | {status} | {details}")
        return status == "PASS"
    except Exception as e:
        print(f"  {label} M={M:>5} | EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    os.environ["AITER_DEBUG_MOE"] = "1"

    print("=" * 100)
    print("Test 1: 16x16 fp8 baseline (n_lane=16, K_align=256)")
    print("=" * 100)
    os.environ["AITER_MOE_WARP32"] = "0"

    # Just one M to keep output readable
    test_kernel("16x16", hidden_raw=2880, inter_raw=360, k_align=256, n_lane=16, M=512)

    print()
    print("=" * 100)
    print("Test 2: 32x32 fp8 minimal (n_lane=32, K_align=128)")
    print("=" * 100)
    os.environ["AITER_MOE_WARP32"] = "1"

    # Smallest possible: K=128 (1 K-block)
    test_kernel("32x32-K128", hidden_raw=128, inter_raw=128, k_align=128, n_lane=32, M=512)

    print()
    print("=" * 100)
    print("Test 3: 32x32 bf16 same dims (M=256 forces bf16 path)")
    print("=" * 100)
    os.environ["AITER_MOE_WARP32"] = "1"
    # M=256 < 512 → bf16 activation, same 32x32 weight layout
    test_kernel("32x32-bf16-K128", hidden_raw=128, inter_raw=128, k_align=128, n_lane=32, M=256)

    print()
    print("=" * 100)
    print("Test 4: 32x32 fp8 production dims")
    print("=" * 100)
    os.environ["AITER_MOE_WARP32"] = "1"
    test_kernel("32x32", hidden_raw=2880, inter_raw=360, k_align=128, n_lane=32, M=512)


if __name__ == "__main__":
    main()
