#!/usr/bin/env python3
"""Detailed row-by-row check for fp8 32x32 kernel."""
import torch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType
from aiter.fused_moe import fused_topk, cktile_moe_stage1, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility.fp4_utils import moe_mxfp4_sort

torch.set_default_device("cuda")
os.environ["AITER_MOE_WARP32"] = "1"

EXPERTS = 2; TOPK = 1; M = 512; hidden = 128; inter = 128
n_lane = 32; block_m = 64

# Weights: all fp4 1.5, scale 1.0
w1_bytes = torch.full((EXPERTS, inter*2, hidden//2), 0x33, dtype=torch.uint8)
w1_scale = torch.full((EXPERTS*inter*2, hidden//32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, True)
w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
w1_s.is_shuffled = True
w1_v = w1_s
if hasattr(torch, "float4_e2m1fn_x2"):
    w1_v = w1_s.view(torch.float4_e2m1fn_x2)
    w1_v.is_shuffled = True

score = torch.zeros((M, EXPERTS), dtype=dtypes.bf16)
score[:, 0] = 1.0
hidden_states = torch.ones((M, hidden), dtype=dtypes.bf16)
topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, EXPERTS, inter*2, moebuf_dtype=dtypes.bf16, block_size=block_m)

a1 = hidden_states.to(dtypes.fp8)
a1_scale = torch.ones([sorted_ids.shape[0], hidden//32], dtype=dtypes.fp8_e8m0, device="cuda")
a1_scale_sorted = moe_mxfp4_sort(a1_scale, sorted_ids, num_valid_ids, M, block_m, n_lane=n_lane)
w2_dummy = torch.zeros((EXPERTS, hidden, inter//2), dtype=torch.uint8)
if hasattr(torch, "float4_e2m1fn_x2"):
    w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

out = cktile_moe_stage1(
    a1, w1_v, w2_dummy, sorted_ids, sorted_expert_ids, num_valid_ids,
    None, TOPK, block_m, a1_scale=a1_scale_sorted,
    w1_scale=w1_scale_s.view(dtypes.fp8_e8m0), activation=ActivationType.Swiglu)
torch.cuda.synchronize()
out_f = out.float()

# Check every row in first 2 blocks
expected = 36864.0
print("Row-by-row results (first 128 rows):")
for r in range(min(128, out_f.shape[0])):
    v = out_f[r, 0, 0].item()
    status = "OK" if abs(v - expected) < 100 else f"WRONG ({v:.0f})"
    if r % 32 == 0 or status != "OK":
        print(f"  Row {r:3d}: {v:10.1f}  {status}")

# Summary per 32-row block
print("\nSummary per 32-row block:")
for block in range(4):
    start = block * 32
    correct = 0
    wrong_val = None
    for r in range(start, start+32):
        if r >= out_f.shape[0]: break
        v = out_f[r, 0, 0].item()
        if abs(v - expected) < 100:
            correct += 1
        else:
            wrong_val = v
    print(f"  Block {block} (rows {start}-{start+31}): {correct}/32 correct" +
          (f", wrong value={wrong_val:.0f}" if wrong_val else ""))

# Find exact boundary in first 32-row block
print("\nFirst block detailed (rows 0-31):")
for r in range(32):
    v = out_f[r, 0, 0].item()
    status = "OK" if abs(v - expected) < 100 else f"WRONG"
    print(f"  Row {r:2d}: {v:10.1f}  {status}")
