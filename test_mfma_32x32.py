#!/usr/bin/env python3
"""Test if the CK 32x32 fp8+fp4 MFMA kernel instance compiles and runs correctly
with synthetic data by using the low-level gemm interface directly.

Instead of going through the full MoE pipeline, use the CK-tile gemm directly
to test just the MFMA computation.
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

def test_identity_routing():
    """Test with identity-like data: very small weights, ones activation.

    If the MFMA computation is correct, the output should be predictable.
    If the data loading is broken, the output will be garbage regardless.
    """
    EXPERTS = 2
    TOPK = 1
    M = 512  # trigger fp8 path

    for warp32, n_lane, k_align, label in [
        ("0", 16, 256, "16x16"),
        ("1", 32, 128, "32x32"),
    ]:
        os.environ["AITER_MOE_WARP32"] = warp32
        block_m = 64 if warp32 == "1" else 32

        hidden = k_align  # minimal K
        inter = k_align

        print(f"\n{'='*70}")
        print(f"{label}: K={hidden}, N={inter*2}")
        print(f"{'='*70}")

        # Create CONSTANT weights: all fp4 = 0.5 (binary 0011)
        # fp4 E2M1: 0011 = sign=0, exp=01, mant=1 → 2^(1-1) * (1 + 0.5) = 1.5
        # packed byte: high nibble=0011, low nibble=0011 → 0x33
        w1_bytes = torch.full((EXPERTS, inter * 2, hidden // 2), 0x33, dtype=torch.uint8)
        w2_bytes = torch.full((EXPERTS, hidden, inter // 2), 0x33, dtype=torch.uint8)

        # Create CONSTANT scales: E8M0 = 127 → scale = 2^0 = 1.0
        w1_scale = torch.full((EXPERTS * inter * 2, hidden // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
        w2_scale = torch.full((EXPERTS * hidden, inter // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)

        # Shuffle
        w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, True)
        w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
        w2_s = shuffle_weight_a16w4(w2_bytes, n_lane, False)
        w2_scale_s = shuffle_scale_a16w4(w2_scale, EXPERTS, False, n_lane=n_lane)
        w1_s.is_shuffled = True
        w2_s.is_shuffled = True

        if hasattr(torch, "float4_e2m1fn_x2"):
            w1_v = w1_s.view(torch.float4_e2m1fn_x2)
            w2_v = w2_s.view(torch.float4_e2m1fn_x2)
            w1_v.is_shuffled = True
            w2_v.is_shuffled = True
        else:
            w1_v = w1_s
            w2_v = w2_s

        # CONSTANT input: all 1.0 in bf16, then convert to fp8
        hidden_states = torch.ones((M, hidden), dtype=dtypes.bf16)

        # Route all tokens to expert 0
        score = torch.zeros((M, EXPERTS), dtype=dtypes.bf16)
        score[:, 0] = 1.0
        topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

        # Expected: each output element = sum of K weight values * 1.0 * scale
        # weight value = 1.5 (fp4 = 0x33, both nibbles = 1.5)
        # scale = 1.0
        # sum over K = K * 1.5 (since activation=1.0, scale=1.0)
        # For K=128: expected = 128 * 1.5 = 192
        # But wait, fp4x2 packs 2 values per byte. K//2 bytes = K values.
        # Input is fp8 = 1.0, so activation scale can be 1.0 too
        expected_per_element = hidden * 1.5
        print(f"  Expected gemm1 per-element (no activation): {expected_per_element}")
        print(f"  (fp8 max = 448, so gemm1 output may overflow for K >= 300)")

        # Run through fused_moe
        out = aiter.fused_moe.fused_moe(
            hidden_states=hidden_states, w1=w1_v, w2=w2_v,
            w1_scale=w1_scale_s, w2_scale=w2_scale_s, a1_scale=None,
            topk_weight=topk_weights, topk_ids=topk_ids,
            quant_type=QuantType.per_1x32, activation=ActivationType.Swiglu,
            expert_mask=None, num_local_tokens=None, dtype=dtypes.bf16,
            hidden_pad=0, intermediate_pad=0,
        )

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        non_nan = out[~torch.isnan(out)]
        print(f"  Output: shape={out.shape} nan={has_nan} inf={has_inf}")
        if len(non_nan) > 0:
            print(f"  Non-NaN: mean={non_nan.abs().mean().item():.4f} max={non_nan.abs().max().item():.4f}")
            print(f"  First 5 values: {out[0, :5].tolist()}")


if __name__ == "__main__":
    os.environ["AITER_DEBUG_MOE"] = "0"
    test_identity_routing()
