#!/usr/bin/env python3
"""Diagnose which rows are wrong in 32x32 fp8 kernel output."""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, cktile_moe_stage1, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

EXPERTS = 2
TOPK = 1
M = 512
DTYPE = dtypes.bf16

os.environ["AITER_MOE_WARP32"] = "1"
os.environ["AITER_DEBUG_MOE"] = "0"

hidden = 128
inter = 128
n_lane = 32
block_m = 64

# Constant weights and scales
w1_bytes = torch.full((EXPERTS, inter * 2, hidden // 2), 0x33, dtype=torch.uint8)
w1_scale = torch.full((EXPERTS * inter * 2, hidden // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
w1_s = shuffle_weight_a16w4(w1_bytes, n_lane, True)
w1_scale_s = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
w1_s.is_shuffled = True

w1_v = w1_s
if hasattr(torch, "float4_e2m1fn_x2"):
    w1_v = w1_s.view(torch.float4_e2m1fn_x2)
    w1_v.is_shuffled = True

# All-ones input
hidden_states = torch.ones((M, hidden), dtype=DTYPE)

# Route ALL tokens to expert 0
score = torch.zeros((M, EXPERTS), dtype=DTYPE)
score[:, 0] = 1.0
topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

# MoE sorting
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, EXPERTS, inter * 2,
    moebuf_dtype=DTYPE, block_size=block_m,
)

print(f"sorted_ids shape: {sorted_ids.shape}")
print(f"num_valid_ids: {num_valid_ids.tolist()}")
print(f"sorted_expert_ids: {sorted_expert_ids.tolist()}")

# Check sorted_ids for padding tokens
sid_cpu = sorted_ids.cpu()
valid = num_valid_ids[0].item()
print(f"sorted_ids[:20]: {sid_cpu[:20].tolist()}")
print(f"sorted_ids[{valid-5}:{valid+5}]: {sid_cpu[max(0,valid-5):valid+5].tolist()}")

a1 = hidden_states.to(dtypes.fp8)
a1_scale = torch.ones([sorted_ids.shape[0], hidden // 32],
                      dtype=dtypes.fp8_e8m0, device=a1.device)

w2_dummy = torch.zeros((EXPERTS, hidden, inter // 2), dtype=torch.uint8)
if hasattr(torch, "float4_e2m1fn_x2"):
    w2_dummy = w2_dummy.view(torch.float4_e2m1fn_x2)

torch.cuda.synchronize()
out = cktile_moe_stage1(
    a1, w1_v, w2_dummy,
    sorted_ids, sorted_expert_ids, num_valid_ids,
    None, TOPK, block_m,
    a1_scale=a1_scale,
    w1_scale=w1_scale_s.view(dtypes.fp8_e8m0),
    activation=ActivationType.Swiglu,
)
torch.cuda.synchronize()

expected = (hidden * 1.5) ** 2  # 192^2 = 36864

out_f = out.float()
# Check per-block statistics
print(f"\nExpected per-element (swiglu): {expected}")
print(f"Output shape: {out.shape}")

for start in range(0, min(sorted_ids.shape[0], 640), 64):
    end = min(start + 64, sorted_ids.shape[0])
    block = out_f[start:end, 0, :]
    nan_count = torch.isnan(block).sum().item()
    non_nan = block[~torch.isnan(block)]
    mean_val = non_nan.abs().mean().item() if len(non_nan) > 0 else float('nan')
    max_val = non_nan.abs().max().item() if len(non_nan) > 0 else float('nan')
    first_val = block[0, 0].item() if block.numel() > 0 else float('nan')
    print(f"  Block [{start:4d}:{end:4d}]: nan={nan_count:5d}/{block.numel()} "
          f"mean={mean_val:12.1f} max={max_val:12.1f} first={first_val}")
