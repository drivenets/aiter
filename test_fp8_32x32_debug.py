#!/usr/bin/env python3
"""Debug: check output values more carefully for the 32x32 fp8 kernel."""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType
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

# Setup weights
w1_bytes = torch.full((EXPERTS, inter * 2, hidden // 2), 0x33, dtype=torch.uint8)
w1_scale = torch.full((EXPERTS * inter * 2, hidden // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, True)
w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
w1_s.is_shuffled = True
w1_v = w1_s
if hasattr(torch, "float4_e2m1fn_x2"):
    w1_v = w1_s.view(torch.float4_e2m1fn_x2)
    w1_v.is_shuffled = True

score = torch.zeros((M, EXPERTS), dtype=dtypes.bf16)
score[:, 0] = 1.0

# Use DISTINCT value per row to trace exactly where data comes from
hidden_states = torch.zeros((M, hidden), dtype=dtypes.bf16)
for i in range(M):
    # Each row gets a distinct fp8-representable value
    # Row i gets value (i % 64) + 1 as bf16, then converted to fp8
    val = float((i % 64) + 1)
    hidden_states[i, :] = val

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

print("=== Row-distinct input test ===")
print(f"sorted_ids[0:4]: {sorted_ids[0:4].tolist()}")
print(f"sorted_ids[32:36]: {sorted_ids[32:36].tolist()}")

# Show what each row's input value is (in fp8)
for r in [0, 1, 15, 31, 32, 33, 47, 63]:
    tok = sorted_ids[r].item() & 0xFFFFFF
    val = a1[tok, 0].float().item()
    print(f"  sorted_pos {r} -> token {tok}, a1 value = {val}")

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

# Expected: GEMM = sum(val * 1.5, 128) = val * 192, swiglu ≈ val^2 * 192^2 = val^2 * 36864
# So output ≈ val^2 * 36864
print("\nOutput (expected = val^2 * 36864):")
for r in [0, 1, 15, 31, 32, 33, 47, 63, 64]:
    v = out_f[r, 0, 0].item()
    tok = sorted_ids[r].item() & 0xFFFFFF if r < sorted_ids.shape[0] else -1
    expected_val = float((tok % 64) + 1) if tok >= 0 else 0
    expected_out = expected_val ** 2 * 36864
    # Infer what input value would produce this output
    if v > 0:
        import math
        inferred_val = math.sqrt(v / 36864)
        print(f"  Row {r}: {v:.0f}  (expected from token {tok}: {expected_out:.0f}, inferred input={inferred_val:.2f})")
    else:
        print(f"  Row {r}: {v:.0f}  (expected from token {tok}: {expected_out:.0f})")
