#!/usr/bin/env python3
"""Test 32x32 fp8 with K=256 (K0=2) vs K=128 (K0=1).
If K0=1 is the degenerate case causing the bug, K=256 should work.
"""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, fused_moe
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

EXPERTS = 4
TOPK = 2
DTYPE = dtypes.bf16
ACTIVATION = ActivationType.Swiglu

def round_up(x, a):
    return ((x + a - 1) // a) * a


def test_k_sizes():
    os.environ["AITER_MOE_WARP32"] = "1"
    n_lane = 32
    k_align = 128

    # Test multiple K sizes for 32x32
    for hidden_raw in [128, 256, 384, 512]:
        hidden = round_up(hidden_raw, k_align)
        inter_raw = 128
        inter = round_up(inter_raw, k_align)
        hidden_pad = hidden - hidden_raw
        inter_pad = inter - inter_raw

        K0 = hidden // 128  # K0 = KPerBlock / (K1 * K2) = hidden / 128

        torch.manual_seed(42)
        w1_raw = torch.randn((EXPERTS, inter * 2, hidden), dtype=DTYPE) / 10
        w2_raw = torch.randn((EXPERTS, hidden, inter), dtype=DTYPE) / 10

        torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
        w1_qt, w1_scale = torch_quant(w1_raw, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = torch_quant(w2_raw, quant_dtype=dtypes.fp4x2)
        w1_qt = w1_qt.view(w1_raw.shape[0], w1_raw.shape[1], w1_raw.shape[2] // 2)
        w2_qt = w2_qt.view(w2_raw.shape[0], w2_raw.shape[1], w2_raw.shape[2] // 2)

        w1_qt = shuffle_weight_a16w4(w1_qt, n_lane, True)
        w1_scale = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
        w2_qt = shuffle_weight_a16w4(w2_qt, n_lane, False)
        w2_scale = shuffle_scale_a16w4(w2_scale, EXPERTS, False, n_lane=n_lane)
        w1_qt.is_shuffled = True
        w2_qt.is_shuffled = True

        w1_v = w1_qt
        w2_v = w2_qt
        if hasattr(torch, "float4_e2m1fn_x2"):
            w1_v = w1_qt.view(torch.float4_e2m1fn_x2)
            w2_v = w2_qt.view(torch.float4_e2m1fn_x2)
            w1_v.is_shuffled = True
            w2_v.is_shuffled = True

        # bf16 path (M=256)
        torch.manual_seed(200)
        M = 256
        hs = torch.randn((M, hidden_raw), dtype=DTYPE)
        score = torch.randn((M, EXPERTS), dtype=DTYPE)
        tw, ti = fused_topk(hs, score, TOPK, True)
        if hidden_pad > 0:
            hs = torch.nn.functional.pad(hs, (0, hidden_pad), value=0.0)
        out_bf16 = fused_moe(hidden_states=hs, w1=w1_v, w2=w2_v,
            w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=None,
            topk_weight=tw, topk_ids=ti, quant_type=QuantType.per_1x32,
            activation=ACTIVATION, expert_mask=None, num_local_tokens=None,
            dtype=DTYPE, hidden_pad=hidden_pad, intermediate_pad=inter_pad)
        if hidden_pad > 0:
            out_bf16 = out_bf16[..., :hidden_raw]
        bf16_mean = out_bf16.abs().mean().item()
        bf16_nan = torch.isnan(out_bf16).any().item()

        # fp8 path (M=512)
        torch.manual_seed(200)
        M = 512
        hs = torch.randn((M, hidden_raw), dtype=DTYPE)
        score = torch.randn((M, EXPERTS), dtype=DTYPE)
        tw, ti = fused_topk(hs, score, TOPK, True)
        if hidden_pad > 0:
            hs = torch.nn.functional.pad(hs, (0, hidden_pad), value=0.0)
        out_fp8 = fused_moe(hidden_states=hs, w1=w1_v, w2=w2_v,
            w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=None,
            topk_weight=tw, topk_ids=ti, quant_type=QuantType.per_1x32,
            activation=ACTIVATION, expert_mask=None, num_local_tokens=None,
            dtype=DTYPE, hidden_pad=hidden_pad, intermediate_pad=inter_pad)
        if hidden_pad > 0:
            out_fp8 = out_fp8[..., :hidden_raw]
        fp8_nan = torch.isnan(out_fp8).any().item()
        non_nan = out_fp8[~torch.isnan(out_fp8)]
        fp8_mean = non_nan.abs().mean().item() if len(non_nan) > 0 else float('nan')
        nan_pct = torch.isnan(out_fp8).sum().item() / out_fp8.numel() * 100

        ratio = fp8_mean / max(bf16_mean, 1e-10) if not (fp8_nan and len(non_nan) == 0) else float('nan')
        status = "OK" if (not fp8_nan and 0.5 < ratio < 2.0) else "FAIL"

        print(f"  K={hidden:>4} (K0={K0}) | bf16: mean={bf16_mean:.4f} nan={bf16_nan} | "
              f"fp8: mean={fp8_mean:.4f} nan={nan_pct:.1f}% | ratio={ratio:.2f} | {status}")


if __name__ == "__main__":
    os.environ["AITER_DEBUG_MOE"] = "0"
    print("32x32 fp8 with varying K (KPerBlock=128, K0 = K/128):")
    test_k_sizes()
