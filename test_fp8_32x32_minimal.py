#!/usr/bin/env python3
"""Minimal test: fp8 activation × fp4 weight GEMM via cktile_moe_stage1.

Tests 16x16 vs 32x32 with same K=256 (padded to both alignments).
Uses all-ones activation scale and all-ones weight scale to eliminate scale bugs.
Uses constant activation (all 1.0 in fp8) and constant weight (all 0x33 fp4 = 1.5).
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
TOPK = 1
M = 512  # >= 512 triggers fp8 path
DTYPE = dtypes.bf16

def test_tile(label, warp32, n_lane, k_align, hidden):
    os.environ["AITER_MOE_WARP32"] = warp32
    block_m = 64 if warp32 == "1" else 32

    inter = k_align  # minimal inter (gate+up = 2*inter)
    print(f"\n{'='*70}")
    print(f"{label}: hidden(K)={hidden}, inter(N/2)={inter}, block_m={block_m}")
    print(f"{'='*70}")

    # CONSTANT weights: all fp4 nibbles = 0x3 = 1.5 in E2M1
    w1_bytes = torch.full((EXPERTS, inter * 2, hidden // 2), 0x33, dtype=torch.uint8)

    # CONSTANT scales: E8M0 = 127 → scale = 2^0 = 1.0
    w1_scale = torch.full((EXPERTS * inter * 2, hidden // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)

    # Shuffle weights
    w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, True)
    w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
    w1_s.is_shuffled = True

    w1_v = w1_s
    if hasattr(torch, "float4_e2m1fn_x2"):
        w1_v = w1_s.view(torch.float4_e2m1fn_x2)
        w1_v.is_shuffled = True

    # CONSTANT activation: all 1.0 in bf16 → will be converted to fp8
    hidden_states = torch.ones((M, hidden), dtype=DTYPE)

    # Route all tokens to expert 0
    score = torch.zeros((M, EXPERTS), dtype=DTYPE)
    score[:, 0] = 1.0
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    # MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, EXPERTS, inter * 2,
        moebuf_dtype=DTYPE, block_size=block_m,
    )

    # Convert to fp8
    a1 = hidden_states.to(dtypes.fp8)

    # All-ones activation scale
    a1_scale = torch.ones([sorted_ids.shape[0], hidden // 32],
                          dtype=dtypes.fp8_e8m0, device=a1.device)

    # Dummy w2
    w2_dummy = torch.zeros((EXPERTS, hidden, inter // 2), dtype=torch.uint8)
    if hasattr(torch, "float4_e2m1fn_x2"):
        w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

    # Expected: each element = sum_k(activation_k * weight_k * a_scale * w_scale)
    # activation_k = 1.0 (fp8), weight_k = 1.5 (fp4 0x3), a_scale = 1.0, w_scale = 1.0
    # sum over K elements = K * 1.0 * 1.5 * 1.0 * 1.0 = K * 1.5
    expected = hidden * 1.5
    print(f"  Expected gemm1 output per element (before swiglu): {expected}")
    print(f"  (fp8 range = [-448, 448], overflow if K*1.5 > 448 → K > 298)")

    torch.cuda.synchronize()
    try:
        out = cktile_moe_stage1(
            a1, w1_v, w2_dummy,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            None, TOPK, block_m,
            a1_scale=a1_scale,
            w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
            activation=ActivationType.Swiglu,
        )
        torch.cuda.synchronize()

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        out_f = out.float()
        non_nan = out_f[~torch.isnan(out_f)]
        print(f"  Output: shape={out.shape} nan={has_nan} inf={has_inf}")
        if len(non_nan) > 0:
            print(f"  Non-NaN: mean={non_nan.abs().mean().item():.4f} max={non_nan.abs().max().item():.4f}")
            print(f"  First row, first 8: {out_f[0, 0, :8].tolist()}")
        if has_nan:
            nan_count = torch.isnan(out_f).sum().item()
            print(f"  NaN count: {nan_count}/{out_f.numel()} ({100*nan_count/out_f.numel():.1f}%)")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    os.environ["AITER_DEBUG_MOE"] = "0"

    # Test with K=256 (safe for both, no fp8 overflow: 256*1.5=384 < 448)
    print("\n" + "#"*70)
    print("# Test 1: K=256 (within fp8 range)")
    print("#"*70)
    test_tile("16x16", "0", 16, 256, hidden=256)
    test_tile("32x32", "1", 32, 128, hidden=256)  # K=256 > 128 align, still fine

    # Test with K=128 (K*1.5=192, well within range)
    print("\n" + "#"*70)
    print("# Test 2: K=128")
    print("#"*70)
    test_tile("16x16", "0", 16, 256, hidden=256)  # padded to 256
    test_tile("32x32", "1", 32, 128, hidden=128)
