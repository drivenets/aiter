#!/usr/bin/env python3
"""Isolate the scale bug in 32x32 fp8+fp4 MoE kernel.

Compare fp8 vs bf16 activation paths on identical weights.
Forces all-ones scales to eliminate scale loading as a variable.
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


def test_scale_bug():
    n_lane = 32
    k_align = 128
    hidden_raw = 128
    inter_raw = 128
    hidden = round_up(hidden_raw, k_align)  # 128
    inter = round_up(inter_raw, k_align)    # 128
    hidden_pad = hidden - hidden_raw
    inter_pad = inter - inter_raw

    os.environ["AITER_MOE_WARP32"] = "1"

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

    print(f"w1_scale stats: shape={w1_scale.shape} dtype={w1_scale.dtype}")
    # Print w1_scale values as E8M0 exponents
    w1s_float = w1_scale.view(torch.uint8).float()
    print(f"  E8M0 raw bytes: min={w1s_float.min().item():.0f} max={w1s_float.max().item():.0f} mean={w1s_float.mean().item():.1f}")
    # Convert E8M0 to actual scale: 2^(val - 127)
    print(f"  As power: 2^({w1s_float.min().item()-127:.0f}) to 2^({w1s_float.max().item()-127:.0f})")

    # Shuffle
    w1_qt = shuffle_weight_a16w4(w1_qt, n_lane, True)
    w1_scale = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
    w2_qt = shuffle_weight_a16w4(w2_qt, n_lane, False)
    w2_scale = shuffle_scale_a16w4(w2_scale, EXPERTS, False, n_lane=n_lane)
    w1_qt.is_shuffled = True
    w2_qt.is_shuffled = True

    # Test 1: bf16 path (M=256 < 512)
    print("\n" + "="*80)
    print("Test 1: bf16 activation (M=256), 32x32 weight layout, REAL scales")
    print("="*80)
    torch.manual_seed(200)
    M = 256
    hidden_states = torch.randn((M, hidden_raw), dtype=DTYPE)
    score = torch.randn((M, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
    if hidden_pad > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, hidden_pad), value=0.0)

    w1_v = w1_qt
    w2_v = w2_qt
    if hasattr(torch, "float4_e2m1fn_x2"):
        w1_v = w1_qt.view(torch.float4_e2m1fn_x2)
        w2_v = w2_qt.view(torch.float4_e2m1fn_x2)
        w1_v.is_shuffled = True
        w2_v.is_shuffled = True

    out_bf16 = fused_moe(
        hidden_states=hidden_states, w1=w1_v, w2=w2_v,
        w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=None,
        topk_weight=topk_weights, topk_ids=topk_ids,
        quant_type=QuantType.per_1x32, activation=ACTIVATION,
        expert_mask=None, num_local_tokens=None, dtype=DTYPE,
        hidden_pad=hidden_pad, intermediate_pad=inter_pad,
    )
    if hidden_pad > 0:
        out_bf16 = out_bf16[..., :hidden_raw]
    print(f"  bf16 output: mean={out_bf16.abs().mean().item():.6f} max={out_bf16.abs().max().item():.6f} nan={torch.isnan(out_bf16).any().item()}")

    # Test 2: fp8 path (M=512 >= 512)
    print("\n" + "="*80)
    print("Test 2: fp8 activation (M=512), 32x32 weight layout, REAL scales")
    print("="*80)
    torch.manual_seed(200)
    M = 512
    hidden_states = torch.randn((M, hidden_raw), dtype=DTYPE)
    score = torch.randn((M, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
    if hidden_pad > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, hidden_pad), value=0.0)

    out_fp8 = fused_moe(
        hidden_states=hidden_states, w1=w1_v, w2=w2_v,
        w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=None,
        topk_weight=topk_weights, topk_ids=topk_ids,
        quant_type=QuantType.per_1x32, activation=ACTIVATION,
        expert_mask=None, num_local_tokens=None, dtype=DTYPE,
        hidden_pad=hidden_pad, intermediate_pad=inter_pad,
    )
    if hidden_pad > 0:
        out_fp8 = out_fp8[..., :hidden_raw]
    print(f"  fp8 output: mean={out_fp8.abs().mean().item():.6f} max={out_fp8.abs().max().item():.6f} nan={torch.isnan(out_fp8).any().item()}")

    # Test 3: fp8 path with ALL scales forced to 1.0 (E8M0 = 127)
    print("\n" + "="*80)
    print("Test 3: fp8 activation (M=512), ALL scales forced to 1.0")
    print("="*80)
    w1_scale_ones = torch.full_like(w1_scale.view(torch.uint8), 127).view(w1_scale.dtype)
    w2_scale_ones = torch.full_like(w2_scale.view(torch.uint8), 127).view(w2_scale.dtype)

    torch.manual_seed(200)
    M = 512
    hidden_states = torch.randn((M, hidden_raw), dtype=DTYPE)
    score = torch.randn((M, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
    if hidden_pad > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, hidden_pad), value=0.0)

    out_fp8_ones = fused_moe(
        hidden_states=hidden_states, w1=w1_v, w2=w2_v,
        w1_scale=w1_scale_ones, w2_scale=w2_scale_ones, a1_scale=None,
        topk_weight=topk_weights, topk_ids=topk_ids,
        quant_type=QuantType.per_1x32, activation=ACTIVATION,
        expert_mask=None, num_local_tokens=None, dtype=DTYPE,
        hidden_pad=hidden_pad, intermediate_pad=inter_pad,
    )
    if hidden_pad > 0:
        out_fp8_ones = out_fp8_ones[..., :hidden_raw]
    print(f"  fp8 (ones scale) output: mean={out_fp8_ones.abs().mean().item():.6f} max={out_fp8_ones.abs().max().item():.6f} nan={torch.isnan(out_fp8_ones).any().item()}")

    # Test 4: bf16 path with ALL scales forced to 1.0
    print("\n" + "="*80)
    print("Test 4: bf16 activation (M=256), ALL scales forced to 1.0")
    print("="*80)
    torch.manual_seed(200)
    M = 256
    hidden_states = torch.randn((M, hidden_raw), dtype=DTYPE)
    score = torch.randn((M, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
    if hidden_pad > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, hidden_pad), value=0.0)

    out_bf16_ones = fused_moe(
        hidden_states=hidden_states, w1=w1_v, w2=w2_v,
        w1_scale=w1_scale_ones, w2_scale=w2_scale_ones, a1_scale=None,
        topk_weight=topk_weights, topk_ids=topk_ids,
        quant_type=QuantType.per_1x32, activation=ACTIVATION,
        expert_mask=None, num_local_tokens=None, dtype=DTYPE,
        hidden_pad=hidden_pad, intermediate_pad=inter_pad,
    )
    if hidden_pad > 0:
        out_bf16_ones = out_bf16_ones[..., :hidden_raw]
    print(f"  bf16 (ones scale) output: mean={out_bf16_ones.abs().mean().item():.6f} max={out_bf16_ones.abs().max().item():.6f} nan={torch.isnan(out_bf16_ones).any().item()}")

    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"  bf16 real scales:  mean={out_bf16.abs().mean().item():.6f}")
    print(f"  fp8  real scales:  mean={out_fp8.abs().mean().item():.6f}  ratio to bf16: {out_fp8.abs().mean().item() / max(out_bf16.abs().mean().item(), 1e-10):.1f}x")
    print(f"  fp8  ones scales:  mean={out_fp8_ones.abs().mean().item():.6f}")
    print(f"  bf16 ones scales:  mean={out_bf16_ones.abs().mean().item():.6f}")
    if not torch.isnan(out_fp8_ones).any():
        print(f"  fp8/bf16 ones ratio: {out_fp8_ones.abs().mean().item() / max(out_bf16_ones.abs().mean().item(), 1e-10):.1f}x")


if __name__ == "__main__":
    os.environ["AITER_DEBUG_MOE"] = "0"  # disable verbose debug for this test
    test_scale_bug()
