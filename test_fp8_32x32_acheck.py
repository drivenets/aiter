#!/usr/bin/env python3
"""Check if A data for rows 32-63 is zero by using non-uniform input."""
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

EXPERTS = 2
TOPK = 1
M = 512
hidden = 128
inter = 128
n_lane = 32
block_m = 64

# Setup weights: all 0x33 fp4 (value 1.5 in each nibble)
w1_bytes = torch.full((EXPERTS, inter * 2, hidden // 2), 0x33, dtype=torch.uint8)
w1_scale = torch.full((EXPERTS * inter * 2, hidden // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, True)
w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
w1_s.is_shuffled = True
w1_v = w1_s
if hasattr(torch, "float4_e2m1fn_x2"):
    w1_v = w1_s.view(torch.float4_e2m1fn_x2)
    w1_v.is_shuffled = True

# Score: all tokens to expert 0
score = torch.zeros((M, EXPERTS), dtype=dtypes.bf16)
score[:, 0] = 1.0

# Test 1: All-1.0 input (baseline)
print("=== Test 1: Uniform input (all 1.0) ===")
hidden_states = torch.ones((M, hidden), dtype=dtypes.bf16)
topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, EXPERTS, inter * 2,
    moebuf_dtype=dtypes.bf16, block_size=block_m,
)
a1 = hidden_states.to(dtypes.fp8)
a1_scale = torch.ones([sorted_ids.shape[0], hidden // 32], dtype=dtypes.fp8_e8m0, device="cuda")
a1_scale_sorted = moe_mxfp4_sort(a1_scale, sorted_ids, num_valid_ids, M, block_m, n_lane=n_lane)
w2_dummy = torch.zeros((EXPERTS, hidden, inter // 2), dtype=torch.uint8)
if hasattr(torch, "float4_e2m1fn_x2"):
    w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

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
for r in [0, 15, 31, 32, 33, 47, 63, 64]:
    print(f"  Row {r}: {out_f[r, 0, 0].item():.1f}")

# Test 2: Row-dependent input to check if kernel reads correct rows
print("\n=== Test 2: Row-dependent input (row i has all fp8(2.0) if i>=32, else fp8(1.0)) ===")
hidden_states2 = torch.ones((M, hidden), dtype=dtypes.bf16)
# Make rows that will map to sorted positions 32-63 have value 2.0
# With all tokens going to expert 0 and topk=1, sorted_ids maps
# sorted_position -> original_token_id directly.
# So sorted position 32 -> token 32 (approximately)
# Let's check sorted_ids
print(f"  sorted_ids[0:5]: {sorted_ids[0:5].tolist()}")
print(f"  sorted_ids[30:35]: {sorted_ids[30:35].tolist()}")
print(f"  sorted_ids[60:65]: {sorted_ids[60:65].tolist()}")

# Set tokens that map to sorted positions 32-63 to have value 2.0
for i in range(32, 64):
    tok_id = sorted_ids[i].item() & 0xFFFFFF  # strip topk bits
    hidden_states2[tok_id, :] = 2.0

a2 = hidden_states2.to(dtypes.fp8)
print(f"  a2[0, 0] = {a2[0, 0].float().item()}, a2[{(sorted_ids[32].item() & 0xFFFFFF)}, 0] = {a2[(sorted_ids[32].item() & 0xFFFFFF), 0].float().item()}")

out2 = cktile_moe_stage1(
    a2, w1_v, w2_dummy,
    sorted_ids, sorted_expert_ids, num_valid_ids,
    None, TOPK, block_m,
    a1_scale=a1_scale_sorted,
    w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
    activation=ActivationType.Swiglu,
)
torch.cuda.synchronize()
out2_f = out2.float()

# With input=1.0 (fp8): GEMM = sum(1.0 * 1.5, 128 times) = 192
# With input=2.0 (fp8): GEMM = sum(2.0 * 1.5, 128 times) = 384
# Expected swiglu:
#   input=1.0: silu(192) * 192 ≈ 192 * 192 = 36864
#   input=2.0: silu(384) * 384 ≈ 384 * 384 = 147456
print("\n  Expected:")
print(f"    input=1.0: silu(192)*192 ≈ 36864")
print(f"    input=2.0: silu(384)*384 ≈ 147456")

for r in [0, 15, 31, 32, 33, 47, 63, 64]:
    v = out2_f[r, 0, 0].item()
    label = ""
    if abs(v - 36864) < 100:
        label = " (matches input=1.0)"
    elif abs(v - 147456) < 1000:
        label = " (matches input=2.0)"
    elif abs(v) < 1e-6:
        label = " (ZERO!)"
    print(f"  Row {r}: {v:.1f}{label}")

# Test 3: Use ZERO input to see if rows 0-31 are affected
print("\n=== Test 3: Zero input for rows 0-31, ones for 32-63 ===")
hidden_states3 = torch.zeros((M, hidden), dtype=dtypes.bf16)
for i in range(32, 64):
    tok_id = sorted_ids[i].item() & 0xFFFFFF
    hidden_states3[tok_id, :] = 1.0

a3 = hidden_states3.to(dtypes.fp8)
out3 = cktile_moe_stage1(
    a3, w1_v, w2_dummy,
    sorted_ids, sorted_expert_ids, num_valid_ids,
    None, TOPK, block_m,
    a1_scale=a1_scale_sorted,
    w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
    activation=ActivationType.Swiglu,
)
torch.cuda.synchronize()
out3_f = out3.float()
print(f"  Expected: rows 0-31 should be 0.0, rows 32-63 should be 36864 (if kernel reads correctly)")
for r in [0, 15, 31, 32, 33, 47, 63, 64]:
    print(f"  Row {r}: {out3_f[r, 0, 0].item():.1f}")
