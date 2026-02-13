#!/usr/bin/env python3
"""Diagnose exact wrong values in 32x32 fp8 kernel output."""
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

# Run with Silu (no gate split) to isolate GEMM output
torch.cuda.synchronize()
out = cktile_moe_stage1(
    a1, w1_v, w2_dummy,
    sorted_ids, sorted_expert_ids, num_valid_ids,
    None, TOPK, block_m,
    a1_scale=a1_scale,
    w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
    activation=ActivationType.Silu,
)
torch.cuda.synchronize()

out_f = out.float()
expected_gemm = hidden * 1.5  # 192.0
# Silu output = silu(192) * (not applicable for Silu-only, different from Swiglu)
# For Silu: output = silu(gemm_result) for each element
# silu(192) = 192 * sigmoid(192) ≈ 192.0

print(f"Output shape: {out.shape}")
print(f"Expected GEMM per element: {expected_gemm}")

# Block 0 analysis (rows 0-63)
block = out_f[0:64, 0, :]
print(f"\nBlock [0:64]: shape={block.shape}")

# Check each row
for r in range(64):
    row = block[r, :]
    nan_count = torch.isnan(row).sum().item()
    if nan_count > 0:
        print(f"  Row {r}: {nan_count} NaN")
        continue
    unique = row.unique()
    if len(unique) == 1 and abs(unique[0].item() - expected_gemm) < 1.0:
        continue  # correct
    wrong = (row - expected_gemm).abs() > 1.0
    if wrong.any():
        wrong_cols = torch.where(wrong)[0]
        vals = row[wrong_cols[:10]]
        print(f"  Row {r}: {wrong.sum().item()} wrong cols. First wrong cols: {wrong_cols[:10].tolist()}")
        print(f"    Values: {vals.tolist()}")
        print(f"    Expected: {expected_gemm}")

# Also check block 1 (rows 64-127) for NaN pattern
block1 = out_f[64:128, 0, :]
nan_per_row = torch.isnan(block1).sum(dim=1)
nan_rows = torch.where(nan_per_row > 0)[0]
print(f"\nBlock [64:128] NaN rows within block: {nan_rows.tolist()}")

# For non-NaN rows in block 1, check values
ok_rows_1 = torch.where(nan_per_row == 0)[0]
for r in ok_rows_1[:5].tolist():
    row = block1[r, :]
    wrong = (row - expected_gemm).abs() > 1.0
    if wrong.any():
        wrong_cols = torch.where(wrong)[0][:5]
        print(f"  Row {64+r}: wrong={wrong.sum().item()} cols, vals={row[wrong_cols].tolist()}")

print("\n--- Also run with ActivationType.No to see raw GEMM ---")
try:
    out_no = cktile_moe_stage1(
        a1, w1_v, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        None, TOPK, block_m,
        a1_scale=a1_scale,
        w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
        activation=ActivationType.No,
    )
    torch.cuda.synchronize()
    out_no_f = out_no.float()
    # Expected raw GEMM = 192.0 for gate columns, 192.0 for up columns (256 total)
    block_no = out_no_f[0:64, 0, :]
    print(f"No-activation output shape: {out_no.shape}")
    print(f"Block [0:64] no-act: first row[:8]={block_no[0, :8].tolist()}")
    nan_no = torch.isnan(block_no).sum().item()
    print(f"Block [0:64] no-act NaN: {nan_no}/{block_no.numel()}")

    # Find wrong values
    for r in range(min(64, block_no.shape[0])):
        row = block_no[r, :]
        wrong = (row - expected_gemm).abs() > 1.0
        if wrong.any() and not torch.isnan(row).any():
            wrong_cols = torch.where(wrong)[0][:5]
            print(f"  Row {r} no-act: wrong={wrong.sum().item()} vals={row[wrong_cols].tolist()}")
            break

    # NaN pattern in block 1
    block1_no = out_no_f[64:128, 0, :]
    nan_per_row_no = torch.isnan(block1_no).sum(dim=1)
    nan_rows_no = torch.where(nan_per_row_no > 0)[0]
    print(f"Block [64:128] no-act NaN rows: {nan_rows_no.tolist()}")
except Exception as e:
    print(f"No-activation failed: {e}")
