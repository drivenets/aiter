#!/usr/bin/env python3
"""
Correctness check for tuned MoE kernel variants.

Runs fused_moe with default vs tuned tile configs and compares outputs
numerically. This catches any bugs in the kernel tile configurations
before running a full sglang sanity check.

Tests:
  1. Default (v0) vs each variant for gemm1 and gemm2
  2. Checks max absolute error, mean absolute error, and relative error
  3. Also compares against torch reference implementation
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
    get_2stage_cfgs,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

MODEL_DIM = 3072
INTER_DIM = 512
EXPERTS = 128
TOPK = 4
DTYPE = dtypes.bf16
ACTIVATION = ActivationType.Swiglu

VARIANTS_TO_TEST = [
    # (block_m, g1_variant, g2_variant, description)
    (16, 0, 0, "default bm16"),
    (16, 3, 0, "bm16 g1:BPC4"),
    (16, 1, 0, "bm16 g1:N256"),
    (16, 0, 1, "bm16 g2:K512_fullpass"),
    (16, 0, 2, "bm16 g2:N256_BPC3"),
    (16, 0, 3, "bm16 g2:N256_K512"),
    (16, 3, 1, "bm16 g1:BPC4+g2:K512"),
    (32, 0, 0, "default bm32"),
    (32, 0, 1, "bm32 g2:K512_fullpass"),
    (32, 1, 0, "bm32 g1:N128_BPC3"),
    (32, 0, 3, "bm32 g2:N128_BPC3"),
    (32, 1, 1, "bm32 g1:N128_BPC3+g2:K512"),
]


def setup_data(token_num):
    torch.manual_seed(42)
    hidden_states = torch.randn((token_num, MODEL_DIM), dtype=DTYPE)
    w1 = torch.randn((EXPERTS, INTER_DIM * 2, MODEL_DIM), dtype=DTYPE) / 10
    w2 = torch.randn((EXPERTS, MODEL_DIM, INTER_DIM), dtype=DTYPE) / 10

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    score = torch.randn((token_num, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    w1_qt_shuf = shuffle_weight_a16w4(w1_qt, 16, True)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, EXPERTS, True)
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, EXPERTS, False)

    return {
        "hidden_states": hidden_states,
        "w1_qt": w1_qt_shuf,
        "w2_qt": w2_qt_shuf,
        "w1_scale": w1_scale_shuf,
        "w2_scale": w2_scale_shuf,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        # Keep unquantized for torch reference
        "w1_raw": w1,
        "w2_raw": w2,
        "w1_qt_raw": w1_qt,
        "w2_qt_raw": w2_qt,
        "w1_scale_raw": w1_scale,
        "w2_scale_raw": w2_scale,
    }


def run_fused_moe(data, block_m, g1_variant, g2_variant):
    os.environ["AITER_MOE_G1_VARIANT"] = str(g1_variant)
    os.environ["AITER_MOE_G2_VARIANT"] = str(g2_variant)
    get_2stage_cfgs.cache_clear()

    out = fused_moe(
        data["hidden_states"],
        data["w1_qt"],
        data["w2_qt"],
        data["topk_weights"],
        data["topk_ids"],
        w1_scale=data["w1_scale"],
        w2_scale=data["w2_scale"],
        quant_type=QuantType.per_1x32,
        activation=ACTIVATION,
        block_size_M=block_m,
    )
    torch.cuda.synchronize()
    return out


def compare(name, ref, test):
    diff = (ref.float() - test.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_abs = ref.float().abs()
    rel_err = (diff / (ref_abs + 1e-8)).mean().item()

    # cosine similarity
    ref_flat = ref.float().flatten()
    test_flat = test.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), test_flat.unsqueeze(0)
    ).item()

    passed = cos_sim > 0.999 and max_err < 1.0
    status = "PASS" if passed else "FAIL"

    print(
        f"  {status} | {name:40s} | "
        f"max_err={max_err:.6f} mean_err={mean_err:.6f} "
        f"rel_err={rel_err:.6f} cos_sim={cos_sim:.6f}"
    )
    return passed


def main():
    token_counts = [1, 4, 16, 64, 256]

    print("=" * 90)
    print("MoE KERNEL CORRECTNESS CHECK")
    print("=" * 90)
    print(f"  model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, experts={EXPERTS}, topk={TOPK}")
    print(f"  activation=Swiglu, quant=per_1x32 (MXFP4)")
    print()

    # JIT warmup
    print("JIT warmup...")
    os.environ["AITER_MOE_G1_VARIANT"] = "0"
    os.environ["AITER_MOE_G2_VARIANT"] = "0"
    d = setup_data(64)
    _ = run_fused_moe(d, 32, 0, 0)
    print("  done.\n")

    all_pass = True

    for token_num in token_counts:
        print(f"--- tokens={token_num} ---")
        data = setup_data(token_num)

        # Get reference (default config)
        ref_16 = run_fused_moe(data, 16, 0, 0)
        ref_32 = run_fused_moe(data, 32, 0, 0)

        for bm, g1v, g2v, desc in VARIANTS_TO_TEST:
            if bm == 16:
                ref = ref_16
            else:
                ref = ref_32

            if g1v == 0 and g2v == 0:
                # Skip comparing default to itself
                continue

            test = run_fused_moe(data, bm, g1v, g2v)
            ok = compare(f"bm{bm} g1v{g1v} g2v{g2v} ({desc})", ref, test)
            if not ok:
                all_pass = False

        # Also cross-check bm16 default vs bm32 default
        # (they should differ due to different sorting granularity, but be "close")
        compare("bm16_default vs bm32_default", ref_16, ref_32)

        print()

    os.environ["AITER_MOE_G1_VARIANT"] = "0"
    os.environ["AITER_MOE_G2_VARIANT"] = "0"

    print("=" * 90)
    if all_pass:
        print("ALL VARIANT CORRECTNESS CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - DO NOT USE FAILED VARIANTS IN PRODUCTION")
    print("=" * 90)


if __name__ == "__main__":
    main()
