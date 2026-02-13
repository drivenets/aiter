#!/usr/bin/env python3
"""Test 32x32 fp8 with different activations and compare against 16x16."""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, cktile_moe_stage1, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")
os.environ["AITER_DEBUG_MOE"] = "0"

EXPERTS = 2
TOPK = 1
M = 512
DTYPE = dtypes.bf16

def run_stage1(label, warp32, n_lane, hidden, inter, activation):
    os.environ["AITER_MOE_WARP32"] = warp32
    block_m = 64 if warp32 == "1" else 32

    # Constant weights
    N_out = inter * 2 if activation == ActivationType.Swiglu else inter
    w1_bytes = torch.full((EXPERTS, N_out, hidden // 2), 0x33, dtype=torch.uint8)
    w1_scale = torch.full((EXPERTS * N_out, hidden // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, activation == ActivationType.Swiglu)
    w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, activation == ActivationType.Swiglu, n_lane=n_lane)
    w1_s.is_shuffled = True
    w1_v = w1_s
    if hasattr(torch, "float4_e2m1fn_x2"):
        w1_v = w1_s.view(torch.float4_e2m1fn_x2)
        w1_v.is_shuffled = True

    hidden_states = torch.ones((M, hidden), dtype=DTYPE)
    score = torch.zeros((M, EXPERTS), dtype=DTYPE)
    score[:, 0] = 1.0
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, EXPERTS, N_out,
        moebuf_dtype=DTYPE, block_size=block_m,
    )

    a1 = hidden_states.to(dtypes.fp8)
    a1_scale = torch.ones([sorted_ids.shape[0], hidden // 32],
                          dtype=dtypes.fp8_e8m0, device=a1.device)

    inter_for_w2 = inter if activation == ActivationType.Swiglu else N_out
    w2_dummy = torch.zeros((EXPERTS, hidden, inter_for_w2 // 2), dtype=torch.uint8)
    if hasattr(torch, "float4_e2m1fn_x2"):
        w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

    torch.cuda.synchronize()
    out = cktile_moe_stage1(
        a1, w1_v, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        None, TOPK, block_m,
        a1_scale=a1_scale,
        w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
        activation=activation,
    )
    torch.cuda.synchronize()

    out_f = out.float()
    has_nan = torch.isnan(out_f).any().item()
    nan_count = torch.isnan(out_f).sum().item()
    non_nan = out_f[~torch.isnan(out_f)]
    mean = non_nan.abs().mean().item() if len(non_nan) > 0 else float('nan')
    first = out_f[0, 0, :4].tolist()
    print(f"  {label}: nan={nan_count}/{out_f.numel()} mean={mean:.2f} first={first}")
    return out_f


print("=== Swiglu activation ===")
run_stage1("16x16", "0", 16, 256, 256, ActivationType.Swiglu)
run_stage1("32x32", "1", 32, 128, 128, ActivationType.Swiglu)

print("\n=== Silu activation (no gate split) ===")
run_stage1("16x16", "0", 16, 256, 256, ActivationType.Silu)
run_stage1("32x32", "1", 32, 128, 128, ActivationType.Silu)

print("\n=== No activation ===")
try:
    run_stage1("16x16", "0", 16, 256, 256, ActivationType.No)
    run_stage1("32x32", "1", 32, 128, 128, ActivationType.No)
except Exception as e:
    print(f"  No activation failed: {e}")
