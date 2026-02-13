#!/usr/bin/env python3
"""Dump sorted scale layout to find where zeros are."""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.utility.fp4_utils import moe_mxfp4_sort

torch.set_default_device("cuda")

os.environ["AITER_MOE_WARP32"] = "1"

EXPERTS = 2
TOPK = 1
M = 512
hidden = 128
block_m = 64
n_lane = 32

hidden_states = torch.ones((M, hidden), dtype=dtypes.bf16)
score = torch.zeros((M, EXPERTS), dtype=dtypes.bf16)
score[:, 0] = 1.0
topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, EXPERTS, 256,  # inter*2 for swiglu
    moebuf_dtype=dtypes.bf16, block_size=block_m,
)

print(f"sorted_ids shape: {sorted_ids.shape}")
print(f"num_valid_ids: {num_valid_ids.tolist()}")

# Create uniform scale
a1_scale = torch.ones([sorted_ids.shape[0], hidden // 32],
                      dtype=dtypes.fp8_e8m0, device="cuda")
print(f"a1_scale shape: {a1_scale.shape}, hidden//32={hidden//32}")
print(f"  a1_scale bytes: all {a1_scale.view(torch.uint8).unique().tolist()}")

# Sort for 32x32
a1_scale_sorted_32 = moe_mxfp4_sort(a1_scale, sorted_ids, num_valid_ids, M, block_m, n_lane=32)
print(f"\nSorted scale (n_lane=32) shape: {a1_scale_sorted_32.shape}")

# Examine as int32
s32 = a1_scale_sorted_32.view(torch.int32)
print(f"  int32 shape: {s32.shape}")

# Check block 0 (first 32 int32 rows — BLOCK_SIZE_M_u32=32)
# For n_lane=32: BLOCK_SIZE_M_u32=32, BLOCK_SIZE_N_u32=2
# So block 0 scale has 32 rows × 2 cols = 64 int32 values
block_size_m_u32 = 32
block_size_n_u32 = 2

for block in range(2):  # first 2 blocks
    start = block * block_size_m_u32
    end = start + block_size_m_u32
    block_data = s32[start:end, :]
    zero_count = (block_data == 0).sum().item()
    nonzero_count = (block_data != 0).sum().item()
    print(f"\n  Block {block} (int32 rows {start}-{end-1}):")
    print(f"    zero int32: {zero_count}, nonzero: {nonzero_count}")

    # Show first 8 rows
    for r in range(min(8, block_data.shape[0])):
        vals = [hex(v) for v in block_data[r].tolist()]
        print(f"    row {start+r}: {vals}")

    # Find zero positions
    zero_positions = torch.where(block_data == 0)
    if len(zero_positions[0]) > 0:
        print(f"    Zero positions (row, col): {list(zip(zero_positions[0][:10].tolist(), zero_positions[1][:10].tolist()))}")

# Now sort for 16x16 for comparison
a1_scale_sorted_16 = moe_mxfp4_sort(a1_scale, sorted_ids, num_valid_ids, M, 32, n_lane=16)
print(f"\nSorted scale (n_lane=16) shape: {a1_scale_sorted_16.shape}")
s16 = a1_scale_sorted_16.view(torch.int32)

# For n_lane=16: BLOCK_SIZE_M_u32=16, BLOCK_SIZE_N_u32=4
for block in range(2):
    start = block * 16
    end = start + 16
    block_data = s16[start:end, :]
    zero_count = (block_data == 0).sum().item()
    print(f"\n  Block {block} (int32 rows {start}-{end-1}):")
    print(f"    zero int32: {zero_count}")
    for r in range(min(4, block_data.shape[0])):
        vals = [hex(v) for v in block_data[r].tolist()]
        print(f"    row {start+r}: {vals}")
