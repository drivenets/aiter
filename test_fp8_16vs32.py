#!/usr/bin/env python3
"""Compare 16x16 vs 32x32 fp8 kernel output."""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, cktile_moe_stage1, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility.fp4_utils import moe_mxfp4_sort

torch.set_default_device("cuda")

EXPERTS = 2
TOPK = 1
M = 512
hidden = 128
inter = 128

def run_test(n_lane, label):
    block_m = 64 if n_lane == 32 else 32
    if n_lane == 32:
        os.environ["AITER_MOE_WARP32"] = "1"
    else:
        os.environ.pop("AITER_MOE_WARP32", None)

    w1_bytes = torch.full((EXPERTS, inter * 2, hidden // 2), 0x33, dtype=torch.uint8)
    w1_scale = torch.full((EXPERTS * inter * 2, hidden // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, True)
    w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
    w1_s.is_shuffled = True
    w1_v = w1_s
    if hasattr(torch, "float4_e2m1fn_x2"):
        w1_v = w1_s.view(torch.float4_e2m1fn_x2)
        w1_v.is_shuffled = True

    hidden_states = torch.ones((M, hidden), dtype=dtypes.bf16)
    score = torch.zeros((M, EXPERTS), dtype=dtypes.bf16)
    score[:, 0] = 1.0
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, EXPERTS, inter * 2,
        moebuf_dtype=dtypes.bf16, block_size=block_m,
    )

    a1 = hidden_states.to(dtypes.fp8)
    a1_scale = torch.ones([sorted_ids.shape[0], hidden // 32],
                          dtype=dtypes.fp8_e8m0, device=a1.device)
    a1_scale_sorted = moe_mxfp4_sort(a1_scale, sorted_ids, num_valid_ids, M, block_m, n_lane=n_lane)

    w2_dummy = torch.zeros((EXPERTS, hidden, inter // 2), dtype=torch.uint8)
    if hasattr(torch, "float4_e2m1fn_x2"):
        w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

    torch.cuda.synchronize()
    out = cktile_moe_stage1(
        a1, w1_v, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        None, TOPK, block_m,
        a1_scale=a1_scale_sorted,
        w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
        activation=ActivationType.Swiglu,
    )
    torch.cuda.synchronize()

    out_f = out.float()
    print(f"\n=== {label} (n_lane={n_lane}, block_m={block_m}) ===")
    print(f"Output shape: {out.shape}")

    # Check first 2 blocks
    for block_start in [0, block_m]:
        block_end = block_start + block_m
        block_data = out_f[block_start:block_end, 0, :]

        # Count zero rows and correct rows
        expected = 36864.0  # 192^2 for swiglu with all-ones input
        zero_rows = []
        correct_rows = []
        nan_rows = []
        for r in range(block_data.shape[0]):
            row = block_data[r]
            if torch.isnan(row).any():
                nan_rows.append(r + block_start)
            elif (row.abs() < 1e-6).all():
                zero_rows.append(r + block_start)
            elif ((row - expected).abs() < 100).all():
                correct_rows.append(r + block_start)

        print(f"  Block [{block_start}:{block_end}]:")
        print(f"    Correct rows: {len(correct_rows)}/{block_data.shape[0]}")
        if zero_rows:
            print(f"    Zero rows: {zero_rows[:10]}{'...' if len(zero_rows) > 10 else ''}")
        if nan_rows:
            print(f"    NaN rows: {nan_rows[:10]}{'...' if len(nan_rows) > 10 else ''}")
        # Show a few sample values
        for r in [0, block_m//2 - 1, block_m//2, block_m - 1]:
            abs_r = block_start + r
            if abs_r < out_f.shape[0]:
                v = out_f[abs_r, 0, 0].item()
                print(f"    Row {abs_r}: {v}")

# Run 16x16 first
run_test(16, "16x16 fp8")
# Run 32x32
run_test(32, "32x32 fp8")
