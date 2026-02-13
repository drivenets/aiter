#!/usr/bin/env python3
"""Find exact NaN pattern in 32x32 fp8 output."""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, cktile_moe_stage1, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

os.environ["AITER_MOE_WARP32"] = "1"
os.environ["AITER_DEBUG_MOE"] = "0"

EXPERTS = 2
TOPK = 1
M = 512
hidden = 128
inter = 128
n_lane = 32
block_m = 64

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
w2_dummy = torch.zeros((EXPERTS, hidden, inter // 2), dtype=torch.uint8)
if hasattr(torch, "float4_e2m1fn_x2"):
    w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

# Test WITHOUT swiglu to see raw gemm1 output
out = cktile_moe_stage1(
    a1, w1_v, w2_dummy,
    sorted_ids, sorted_expert_ids, num_valid_ids,
    None, TOPK, block_m,
    a1_scale=a1_scale,
    w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
    activation=ActivationType.Swiglu,
)
torch.cuda.synchronize()

out_f = out.float()
print(f"Output shape: {out.shape}")
print(f"Expected swiglu: {(128*1.5)**2}")

# Find NaN rows in first block (rows 0-63 of OUTPUT, which are token IDs 0-63)
nan_mask = torch.isnan(out_f[:, 0, :])  # (512, 128)
nan_per_row = nan_mask.sum(dim=1)  # (512,)
nan_rows = torch.where(nan_per_row > 0)[0]
print(f"\nNaN rows (first 30): {nan_rows[:30].tolist()}")
print(f"Total NaN rows: {nan_rows.shape[0]} out of {M}")

# For non-NaN rows, check values
ok_rows = torch.where(nan_per_row == 0)[0]
if len(ok_rows) > 0:
    ok_vals = out_f[ok_rows, 0, :]
    unique_vals = ok_vals.unique()
    print(f"\nNon-NaN rows: {ok_rows.shape[0]}")
    print(f"Unique values in non-NaN rows: {unique_vals[:20].tolist()}")

# For NaN rows, check which COLUMNS have NaN
if len(nan_rows) > 0:
    row = nan_rows[0].item()
    row_data = out_f[row, 0, :]
    nan_cols = torch.where(torch.isnan(row_data))[0]
    ok_cols = torch.where(~torch.isnan(row_data))[0]
    print(f"\nRow {row}: NaN cols ({len(nan_cols)}): {nan_cols[:20].tolist()}")
    if len(ok_cols) > 0:
        print(f"Row {row}: OK cols ({len(ok_cols)}): {ok_cols[:20].tolist()}")
        print(f"Row {row}: OK values: {row_data[ok_cols[:10]].tolist()}")

# Check wrong-value rows (non-NaN but != expected)
expected = (128 * 1.5) ** 2
for r in ok_rows[:5].tolist():
    vals = out_f[r, 0, :]
    wrong = (vals != expected) & (~torch.isnan(vals))
    if wrong.any():
        wrong_cols = torch.where(wrong)[0]
        print(f"\nRow {r}: wrong value cols: {wrong_cols[:10].tolist()}")
        print(f"  Values: {vals[wrong_cols[:10]].tolist()}")
