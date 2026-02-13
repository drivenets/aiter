#!/usr/bin/env python3
"""Test if scale bug causes wrong values in rows 32-63."""
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

os.environ["AITER_MOE_WARP32"] = "1"
os.environ["AITER_DEBUG_MOE"] = "0"

EXPERTS = 2
TOPK = 1
M = 512
hidden = 128
inter = 128
n_lane = 32
block_m = 64

# Setup
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

# Test 1: All-ones scale (all bytes = 0x7F = 127)
print("=== Test 1: Uniform scale_a = 1.0 (all bytes 0x7F) ===")
a1_scale = torch.ones([sorted_ids.shape[0], hidden // 32],
                      dtype=dtypes.fp8_e8m0, device=a1.device)
# Check raw scale bytes
scale_bytes = a1_scale.view(torch.uint8)
print(f"  Raw scale_a shape: {a1_scale.shape}, unique bytes: {scale_bytes.unique().tolist()}")

# Apply moe_mxfp4_sort (which packs 4 bytes into int32)
a1_scale_sorted = moe_mxfp4_sort(a1_scale, sorted_ids, num_valid_ids, M, block_m, n_lane=n_lane)
sorted_bytes = a1_scale_sorted.view(torch.uint8)
print(f"  Sorted scale shape: {a1_scale_sorted.shape}")
print(f"  Sorted unique bytes: {sorted_bytes.unique().tolist()}")

# Check int32 values
sorted_u32 = a1_scale_sorted.view(torch.int32)
print(f"  Sorted int32 unique: {[hex(v) for v in sorted_u32.unique().tolist()]}")

# Test 2: Use DIFFERENT scale for rows 32-63 to verify scale is being read correctly
print("\n=== Test 2: Different scale for imxdl=0 vs imxdl=1 ===")
# For 32x32 with n_lane=32:
# The CK kernel reads scale_a as (M/64 blocks, K/128 blocks, 4 bytes per int32)
# opsel = ikxdl * MXdlPack + imxdl
# opsel 0: imxdl=0, ikxdl=0 -> scale for M rows 0-31, K iter 0
# opsel 1: imxdl=1, ikxdl=0 -> scale for M rows 32-63, K iter 0
# opsel 2: imxdl=0, ikxdl=1 -> scale for M rows 0-31, K iter 1
# opsel 3: imxdl=1, ikxdl=1 -> scale for M rows 32-63, K iter 1

# Create a scale where byte positions matter:
# All 0x7F except byte 1 (opsel=1, imxdl=1) = 0x80 (scale=2^1=2)
a1_scale_test = torch.ones([sorted_ids.shape[0], hidden // 32],
                           dtype=dtypes.fp8_e8m0, device=a1.device)

# Actually, moe_mxfp4_sort packs the scales based on token position.
# Let me check what the sorted scale looks like for uniform vs non-uniform.

# First, let's just verify: with UNIFORM scale, do rows 0-31 and 32-63 get the SAME values?
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
expected = (hidden * 1.5) ** 2  # 192^2 = 36864

# Show values at key rows in block 0
print(f"\n  Expected (swiglu): {expected}")
for r in [0, 7, 15, 31, 32, 33, 38, 39, 40, 43, 44, 47, 55, 63]:
    v = out_f[r, 0, 0].item()
    nan = "NaN" if torch.isnan(out_f[r, 0, :]).any() else ""
    ratio = v / expected if not (torch.isnan(torch.tensor(v)) or v == 0) else 0
    print(f"  Row {r:3d}: val={v:15.1f} ratio={ratio:10.4f} {nan}")

# Test 3: What if we DON'T use moe_mxfp4_sort? Pass raw scale directly.
print("\n=== Test 3: Pass raw a1_scale (no moe_mxfp4_sort) ===")
a1_scale_raw = torch.ones([sorted_ids.shape[0], hidden // 32],
                          dtype=dtypes.fp8_e8m0, device=a1.device)
torch.cuda.synchronize()
out_raw = cktile_moe_stage1(
    a1, w1_v, w2_dummy,
    sorted_ids, sorted_expert_ids, num_valid_ids,
    None, TOPK, block_m,
    a1_scale=a1_scale_raw,  # raw, not sorted
    w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
    activation=ActivationType.Swiglu,
)
torch.cuda.synchronize()

out_raw_f = out_raw.float()
print(f"  Expected: {expected}")
for r in [0, 7, 31, 32, 33, 38, 39, 47, 55, 63]:
    v = out_raw_f[r, 0, 0].item()
    nan = "NaN" if torch.isnan(out_raw_f[r, 0, :]).any() else ""
    print(f"  Row {r:3d}: val={v:15.1f} {nan}")
