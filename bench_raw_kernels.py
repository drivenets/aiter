#!/usr/bin/env python3
"""
Raw CK-tile MoE kernel benchmark for GPT-OSS shapes.

Tests gemm1 and gemm2 kernels INDEPENDENTLY, bypassing fused_moe overhead
(sorting, quantization, scale shuffling). Pre-computes all inputs once,
then benchmarks pure kernel execution time.

Kernel variants selected via env vars:
  AITER_MOE_G1_VARIANT: gemm1 tile variant (0=default)
  AITER_MOE_G2_VARIANT: gemm2 tile variant (0=default)

Gemm1 (N=1024, K=3072):
  bm16: v0=N128_K256_BPC2, v1=N256_K256_BPC2, v2=N512_K256_BPC4, v3=N128_K256_BPC4
  bm32: v0=N256_K256_BPC2, v1=N128_K256_BPC3, v2=N256_K256_BPC3, v3=N256_K256_BPC1

Gemm2 (N=3072, K=512):
  bm16: v0=N128_K256_BPC2, v1=N128_K512_BPC2, v2=N256_K256_BPC3, v3=N256_K512_BPC2
  bm32: v0=N256_K256_BPC2, v1=N256_K512_BPC2, v2=N256_K256_BPC3, v3=N128_K256_BPC3

Usage:
    python bench_raw_kernels.py
    python bench_raw_kernels.py --tokens 1 4 16 64 256 --block-m 16 32
"""

import torch
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    moe_sorting,
    get_2stage_cfgs,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

# GPT-OSS MoE parameters
MODEL_DIM = 3072
INTER_DIM = 512
EXPERTS = 128
TOPK = 4
DTYPE = dtypes.bf16
ACTIVATION = ActivationType.Swiglu

G1_VARIANTS = {
    16: {
        0: "N128_K256_BPC2 (default)",
        1: "N256_K256_BPC2 (wider N)",
        2: "N512_K256_BPC4 (very wide N)",
        3: "N128_K256_BPC4 (max occ)",
    },
    32: {
        0: "N256_K256_BPC2 (default)",
        1: "N128_K256_BPC3 (narrow N)",
        2: "N256_K256_BPC3 (high occ)",
        3: "N256_K256_BPC1 (low occ)",
    },
}

G2_VARIANTS = {
    16: {
        0: "N128_K256_BPC2 (default)",
        1: "N128_K512_BPC2 (full K!)",
        2: "N256_K256_BPC3 (wide+occ)",
        3: "N256_K512_BPC2 (wide+fullK)",
    },
    32: {
        0: "N256_K256_BPC2 (default)",
        1: "N256_K512_BPC2 (full K!)",
        2: "N256_K256_BPC3 (high occ)",
        3: "N128_K256_BPC3 (narrow+occ)",
    },
}


def setup_data(token_num, block_m):
    """Prepare all pre-sorted, pre-quantized data for raw kernel calls."""
    torch.manual_seed(42)

    hidden_states = torch.randn((token_num, MODEL_DIM), dtype=DTYPE)
    w1 = torch.randn((EXPERTS, INTER_DIM * 2, MODEL_DIM), dtype=DTYPE) / 10
    w2 = torch.randn((EXPERTS, MODEL_DIM, INTER_DIM), dtype=DTYPE) / 10

    # Quantize weights to MXFP4
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Shuffle for CK-tile a16w4
    w1_qt = shuffle_weight_a16w4(w1_qt, 16, True)
    w1_scale = shuffle_scale_a16w4(w1_scale, EXPERTS, True)
    w2_qt = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale = shuffle_scale_a16w4(w2_scale, EXPERTS, False)

    # Routing
    score = torch.randn((token_num, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    # Moe sorting (this is what we DON'T want to benchmark)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, EXPERTS, MODEL_DIM, DTYPE, block_m,
    )

    # For a16w4+Swiglu path: a1 = bf16 hidden_states, a1_scale = None
    a1 = hidden_states.to(DTYPE)

    # Stage1 output shape (for stage2 input)
    _, n1, k1 = w1_qt.shape
    _, k2, n2 = w2_qt.shape
    D = n2 * 2  # fp4x2 packing: actual inter_dim = n2 * 2
    a2 = torch.empty((token_num, TOPK, D), dtype=DTYPE, device="cuda")

    # Stage2 output
    moe_out = torch.empty((token_num, MODEL_DIM), dtype=DTYPE, device="cuda")

    return {
        "a1": a1,
        "w1": w1_qt,
        "w2": w2_qt,
        "w1_scale": w1_scale.view(dtypes.fp8_e8m0),
        "w2_scale": w2_scale.view(dtypes.fp8_e8m0),
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "a2": a2,
        "moe_out": moe_out,
        "token_num": token_num,
    }


def bench_stage1(data, block_m, num_warmup=10, num_iters=50):
    """Benchmark raw cktile_moe_stage1 (gemm1) kernel."""
    token_num = data["token_num"]
    w1 = data["w1"]
    _, n1, k1 = w1.shape
    _, k2, n2 = data["w2"].shape
    D = n2 * 2

    for _ in range(num_warmup):
        out = torch.empty((token_num, TOPK, D), dtype=DTYPE, device="cuda")
        aiter.moe_cktile2stages_gemm1(
            data["a1"], w1, out,
            data["sorted_ids"], data["sorted_expert_ids"], data["num_valid_ids"],
            TOPK,
            0,  # n_pad_zeros
            0,  # k_pad_zeros
            None,  # sorted_weights (stage1 no weight)
            None,  # a1_scale (bf16 activation, no scale)
            data["w1_scale"],
            None,  # bias1
            ACTIVATION,
            block_m,
            1,  # split_k
        )
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        out = torch.empty((token_num, TOPK, D), dtype=DTYPE, device="cuda")
        start_events[i].record()
        aiter.moe_cktile2stages_gemm1(
            data["a1"], w1, out,
            data["sorted_ids"], data["sorted_expert_ids"], data["num_valid_ids"],
            TOPK,
            0, 0,
            None, None,
            data["w1_scale"],
            None,
            ACTIVATION,
            block_m, 1,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times_us = sorted(times_us[2:])  # drop first 2
    median_us = times_us[len(times_us) // 2]

    # FLOPS: M_eff * N * K * 2 (multiply-add)
    # gemm1: token*topk effective tokens, N=inter_dim*2=1024, K=model_dim=3072
    m_eff = token_num * TOPK
    flops = m_eff * (INTER_DIM * 2) * MODEL_DIM * 2
    tflops = flops / median_us / 1e6

    return {"us": median_us, "tflops": tflops, "out": out}


def bench_stage2(data, block_m, a2_input=None, num_warmup=10, num_iters=50):
    """Benchmark raw cktile_moe_stage2 (gemm2) kernel."""
    token_num = data["token_num"]

    # Use provided a2 or create dummy
    if a2_input is None:
        a2 = torch.randn((token_num, TOPK, INTER_DIM), dtype=DTYPE, device="cuda")
    else:
        a2 = a2_input

    moe_out = torch.empty((token_num, MODEL_DIM), dtype=DTYPE, device="cuda")

    for _ in range(num_warmup):
        aiter.moe_cktile2stages_gemm2(
            a2, data["w2"], moe_out,
            data["sorted_ids"], data["sorted_expert_ids"], data["num_valid_ids"],
            TOPK,
            0, 0,
            data["sorted_weights"],  # stage2 applies routing weights
            None,  # a2_scale (bf16, no scale)
            data["w2_scale"],
            None,  # bias2
            ACTIVATION,
            block_m,
        )
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        aiter.moe_cktile2stages_gemm2(
            a2, data["w2"], moe_out,
            data["sorted_ids"], data["sorted_expert_ids"], data["num_valid_ids"],
            TOPK,
            0, 0,
            data["sorted_weights"],
            None,
            data["w2_scale"],
            None,
            ACTIVATION,
            block_m,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times_us = sorted(times_us[2:])
    median_us = times_us[len(times_us) // 2]

    m_eff = token_num * TOPK
    flops = m_eff * MODEL_DIM * INTER_DIM * 2
    tflops = flops / median_us / 1e6

    return {"us": median_us, "tflops": tflops, "out": moe_out}


def set_variant(g1=0, g2=0):
    os.environ["AITER_MOE_G1_VARIANT"] = str(g1)
    os.environ["AITER_MOE_G2_VARIANT"] = str(g2)


def main():
    parser = argparse.ArgumentParser(description="Raw CK-tile MoE kernel benchmark")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--block-m", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    # Only test tokens < 512 (a16w4 boundary)
    args.tokens = [t for t in args.tokens if t < 512]

    print("=" * 80)
    print("RAW CK-TILE MOE KERNEL BENCHMARK (GPT-OSS shapes)")
    print("=" * 80)
    print(f"  Gemm1: M=tokens*{TOPK}, N={INTER_DIM*2}, K={MODEL_DIM}")
    print(f"  Gemm2: M=tokens*{TOPK}, N={MODEL_DIM}, K={INTER_DIM}")
    print(f"  Experts={EXPERTS}, topk={TOPK}, activation=Swiglu, MXFP4")
    print(f"  tokens: {args.tokens}, block_m: {args.block_m}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print()

    # JIT warmup: trigger module compilation with default variant
    print("Triggering JIT compilation...")
    set_variant(0, 0)
    safe_data = setup_data(64, 32)
    _ = bench_stage1(safe_data, 32, num_warmup=2, num_iters=3)
    _ = bench_stage2(safe_data, 32, num_warmup=2, num_iters=3)
    print("  JIT compilation done.\n")

    for bm in args.block_m:
        g1_vars = G1_VARIANTS.get(bm, {0: "default"})
        g2_vars = G2_VARIANTS.get(bm, {0: "default"})

        # ---- GEMM1 sweep ----
        print("=" * 80)
        print(f"GEMM1 (stage1) sweep -- block_m={bm}")
        print(f"  Shape per expert: M=tokens*{TOPK}/{EXPERTS}, N={INTER_DIM*2}, K={MODEL_DIM}")
        print("=" * 80)

        # Header
        hdr = f"{'tokens':>6} | {'M_eff':>6} |"
        for v in sorted(g1_vars.keys()):
            tag = G1_VARIANTS[bm][v].split("(")[0].strip()
            hdr += f" v{v}:{tag:>18} |"
        hdr += " best | delta"
        print(hdr)
        print("-" * len(hdr))

        for token_num in args.tokens:
            data = setup_data(token_num, bm)
            m_eff = token_num * TOPK
            line = f"{token_num:>6} | {m_eff:>6} |"
            best_us = float("inf")
            best_v = -1
            default_us = None

            for v in sorted(g1_vars.keys()):
                set_variant(v, 0)
                try:
                    r = bench_stage1(data, bm, args.warmup, args.iters)
                    us = r["us"]
                    tf = r["tflops"]
                    line += f" {us:>7.1f}us {tf:>5.1f}TF |"
                    if v == 0:
                        default_us = us
                    if us < best_us:
                        best_us = us
                        best_v = v
                except Exception as e:
                    line += f" {'FAIL':>18} |"
                    print(f"  [ERR] t={token_num} v={v}: {e}")

            delta = ((default_us - best_us) / default_us * 100) if default_us else 0
            line += f"  v{best_v} | {delta:>+5.1f}%"
            print(line)

        print()

        # ---- GEMM2 sweep ----
        print("=" * 80)
        print(f"GEMM2 (stage2) sweep -- block_m={bm}")
        print(f"  Shape per expert: M=tokens*{TOPK}/{EXPERTS}, N={MODEL_DIM}, K={INTER_DIM}")
        print("=" * 80)

        hdr = f"{'tokens':>6} | {'M_eff':>6} |"
        for v in sorted(g2_vars.keys()):
            tag = G2_VARIANTS[bm][v].split("(")[0].strip()
            hdr += f" v{v}:{tag:>18} |"
        hdr += " best | delta"
        print(hdr)
        print("-" * len(hdr))

        for token_num in args.tokens:
            data = setup_data(token_num, bm)
            m_eff = token_num * TOPK
            line = f"{token_num:>6} | {m_eff:>6} |"
            best_us = float("inf")
            best_v = -1
            default_us = None

            for v in sorted(g2_vars.keys()):
                set_variant(0, v)
                try:
                    r = bench_stage2(data, bm, num_warmup=args.warmup, num_iters=args.iters)
                    us = r["us"]
                    tf = r["tflops"]
                    line += f" {us:>7.1f}us {tf:>5.1f}TF |"
                    if v == 0:
                        default_us = us
                    if us < best_us:
                        best_us = us
                        best_v = v
                except Exception as e:
                    line += f" {'FAIL':>18} |"
                    print(f"  [ERR] t={token_num} v={v}: {e}")

            delta = ((default_us - best_us) / default_us * 100) if default_us else 0
            line += f"  v{best_v} | {delta:>+5.1f}%"
            print(line)

        print()

    set_variant(0, 0)
    print("DONE")


if __name__ == "__main__":
    main()
