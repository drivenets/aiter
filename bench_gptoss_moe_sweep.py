#!/usr/bin/env python3
"""
Sweep benchmark for CK-tile MoE kernel variants with GPT-OSS shapes.

Tests different tile configurations (NPerBlock, KPerBlock, Block_Per_CU)
for each block_m value using env-var-based dispatch:
  - AITER_MOE_G1_VARIANT: selects gemm1 tile variant (0=default)
  - AITER_MOE_G2_VARIANT: selects gemm2 tile variant (0=default)

Variant descriptions:
  Gemm1 (N=1024, K=3072):
    block_m=16:
      v0: N=128 K=256 BPC=2 (default)
      v1: N=256 K=256 BPC=2 (wider N tile, fewer N-blocks: 4 vs 8)
      v2: N=512 K=256 BPC=4 (very wide N, max occupancy)
      v3: N=128 K=256 BPC=4 (same tile, max occupancy)
    block_m=32:
      v0: N=256 K=256 BPC=2 (default)
      v1: N=128 K=256 BPC=3 (narrower N, higher occupancy)
      v2: N=256 K=256 BPC=3 (same tile, higher occupancy)
      v3: N=256 K=128 BPC=2 (smaller K reduction tile, 24 K-iters vs 12)

  Gemm2 (N=3072, K=512):
    block_m=16:
      v0: N=128 K=256 BPC=2 (default, 2 K-iterations)
      v1: N=128 K=512 BPC=2 (FULL K PASS - single K iteration!)
      v2: N=256 K=256 BPC=3 (wider N, higher occupancy)
      v3: N=256 K=512 BPC=2 (wide N + full K pass)
    block_m=32:
      v0: N=256 K=256 BPC=2 (default, 2 K-iterations)
      v1: N=256 K=512 BPC=2 (FULL K PASS)
      v2: N=256 K=256 BPC=3 (higher occupancy)
      v3: N=128 K=256 BPC=3 (narrower N, higher occupancy)

Usage:
    AITER_REBUILD=1 python bench_gptoss_moe_sweep.py
    python bench_gptoss_moe_sweep.py --phase 1  # block_m sweep only
    python bench_gptoss_moe_sweep.py --phase 2  # tile variant sweep
    python bench_gptoss_moe_sweep.py --phase 3  # combined best
"""

import torch
import argparse
import sys
import os
import functools
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    get_block_size_M,
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

# Variant descriptions for pretty-printing
G1_VARIANTS = {
    16: {
        0: "N=128 K=256 BPC=2 (default)",
        1: "N=256 K=256 BPC=2 (wider N)",
        2: "N=512 K=256 BPC=4 (very wide N, max occ)",
        3: "N=128 K=256 BPC=4 (max occupancy)",
    },
    32: {
        0: "N=256 K=256 BPC=2 (default)",
        1: "N=128 K=256 BPC=3 (narrow N, high occ)",
        2: "N=256 K=256 BPC=3 (high occupancy)",
        3: "N=256 K=256 BPC=1 (low occupancy)",
    },
}

G2_VARIANTS = {
    16: {
        0: "N=128 K=256 BPC=2 (default)",
        1: "N=128 K=512 BPC=2 (FULL K PASS!)",
        2: "N=256 K=256 BPC=3 (wider N, high occ)",
        3: "N=256 K=512 BPC=2 (wide N + full K)",
    },
    32: {
        0: "N=256 K=256 BPC=2 (default)",
        1: "N=256 K=512 BPC=2 (FULL K PASS!)",
        2: "N=256 K=256 BPC=3 (high occupancy)",
        3: "N=128 K=256 BPC=3 (narrow N, high occ)",
    },
}


def setup_moe_data(token_num):
    """Create test data matching GPT-OSS MoE shapes with MXFP4 weights."""
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

    return (
        hidden_states,
        w1_qt_shuf,
        w2_qt_shuf,
        w1_scale_shuf,
        w2_scale_shuf,
        topk_weights,
        topk_ids,
    )


def set_variant(g1_variant=0, g2_variant=0):
    """Set the env vars for kernel variant dispatch."""
    os.environ["AITER_MOE_G1_VARIANT"] = str(g1_variant)
    os.environ["AITER_MOE_G2_VARIANT"] = str(g2_variant)


def bench_fused_moe(data, block_m, num_warmup=5, num_iters=20):
    """Benchmark fused_moe with explicit block_size_M."""
    hidden_states, w1_qt, w2_qt, w1_scale, w2_scale, topk_weights, topk_ids = data
    token_num = hidden_states.shape[0]

    # Clear caches
    get_2stage_cfgs.cache_clear()

    # Warmup
    for _ in range(num_warmup):
        out = fused_moe(
            hidden_states, w1_qt, w2_qt, topk_weights, topk_ids,
            w1_scale=w1_scale, w2_scale=w2_scale,
            quant_type=QuantType.per_1x32,
            activation=ACTIVATION,
            block_size_M=block_m,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        out = fused_moe(
            hidden_states, w1_qt, w2_qt, topk_weights, topk_ids,
            w1_scale=w1_scale, w2_scale=w2_scale,
            quant_type=QuantType.per_1x32,
            activation=ACTIVATION,
            block_size_M=block_m,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    if len(times_ms) > 4:
        times_ms = sorted(times_ms[2:])
        median_ms = times_ms[len(times_ms) // 2]
    else:
        median_ms = sorted(times_ms)[len(times_ms) // 2]

    total_us = median_ms * 1000
    m_eff = token_num * TOPK
    flop_g1 = m_eff * (INTER_DIM * 2) * MODEL_DIM * 2
    flop_g2 = m_eff * MODEL_DIM * INTER_DIM * 2
    tflops = (flop_g1 + flop_g2) / total_us / 1e6

    return {"total_us": total_us, "tflops": tflops}


def jit_warmup(block_ms, variants_g1, variants_g2):
    """Trigger JIT compilation for all needed variants."""
    print("=" * 70)
    print("Phase 0: JIT compilation warmup")
    print("=" * 70)

    # First pass: compile with default variant for all block_m values
    for bm in block_ms:
        set_variant(0, 0)
        try:
            safe_tokens = max(bm * 2, 64)
            data = setup_moe_data(safe_tokens)
            _ = fused_moe(
                data[0], data[1], data[2], data[5], data[6],
                w1_scale=data[3], w2_scale=data[4],
                quant_type=QuantType.per_1x32,
                activation=ACTIVATION,
                block_size_M=bm,
            )
            torch.cuda.synchronize()
            print(f"  block_m={bm}, g1=0, g2=0: OK")
        except Exception as e:
            print(f"  block_m={bm}: FAILED - {e}")

    # Compile variant kernels (they share the same module, just different dispatch)
    # The module is already compiled, variants just select different compiled kernels
    for bm in [16, 32]:
        for gv in variants_g1.get(bm, {}).keys():
            if gv == 0:
                continue
            set_variant(gv, 0)
            try:
                safe_tokens = max(bm * 2, 64)
                data = setup_moe_data(safe_tokens)
                get_2stage_cfgs.cache_clear()
                _ = fused_moe(
                    data[0], data[1], data[2], data[5], data[6],
                    w1_scale=data[3], w2_scale=data[4],
                    quant_type=QuantType.per_1x32,
                    activation=ACTIVATION,
                    block_size_M=bm,
                )
                torch.cuda.synchronize()
                print(f"  block_m={bm}, g1={gv}, g2=0: OK")
            except Exception as e:
                print(f"  block_m={bm}, g1={gv}: FAILED - {e}")

        for gv in variants_g2.get(bm, {}).keys():
            if gv == 0:
                continue
            set_variant(0, gv)
            try:
                safe_tokens = max(bm * 2, 64)
                data = setup_moe_data(safe_tokens)
                get_2stage_cfgs.cache_clear()
                _ = fused_moe(
                    data[0], data[1], data[2], data[5], data[6],
                    w1_scale=data[3], w2_scale=data[4],
                    quant_type=QuantType.per_1x32,
                    activation=ACTIVATION,
                    block_size_M=bm,
                )
                torch.cuda.synchronize()
                print(f"  block_m={bm}, g1=0, g2={gv}: OK")
            except Exception as e:
                print(f"  block_m={bm}, g2={gv}: FAILED - {e}")

    set_variant(0, 0)
    print()


def phase1_block_m_sweep(token_counts, block_ms, args):
    """Phase 1: Test different block_m values with default tile configs."""
    print("=" * 70)
    print("Phase 1: Block_m sweep (default tile configs, variant=0)")
    print("=" * 70)

    set_variant(0, 0)

    hdr = f"{'tokens':>8} | {'heur_bm':>7} |"
    for bm in block_ms:
        hdr += f" bm={bm:>3}_us |"
    hdr += " best_bm | speedup"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for token_num in token_counts:
        heuristic_bm = get_block_size_M(token_num, TOPK, EXPERTS, INTER_DIM)
        data = setup_moe_data(token_num)

        line = f"{token_num:>8} | bm={heuristic_bm:>3} |"
        best_us = float("inf")
        best_bm = -1
        heuristic_us = None
        row = {"tokens": token_num, "heuristic_bm": heuristic_bm}

        for bm in block_ms:
            try:
                get_2stage_cfgs.cache_clear()
                r = bench_fused_moe(data, bm, args.warmup, args.iters)
                total_us = r["total_us"]
                row[f"bm{bm}_us"] = total_us
                line += f" {total_us:>8.1f} |"
                if total_us < best_us:
                    best_us = total_us
                    best_bm = bm
                if bm == heuristic_bm:
                    heuristic_us = total_us
            except Exception as e:
                line += f" {'FAIL':>8} |"
                print(f"  [WARN] tokens={token_num}, bm={bm}: {e}")

        speedup = heuristic_us / best_us if heuristic_us and best_us < float("inf") else 0
        line += f" bm={best_bm:>3} | {speedup:.2f}x"
        print(line)
        results.append(row)

    return results


def phase2_variant_sweep(token_counts, args):
    """Phase 2: Sweep tile variants for gemm1 and gemm2 independently."""
    print()
    print("=" * 70)
    print("Phase 2: Tile variant sweep (gemm1 & gemm2 independently)")
    print("=" * 70)

    all_results = {}

    for bm in [16, 32]:
        print(f"\n--- block_m={bm}: Gemm1 variants (G2=default) ---")
        g1_descs = G1_VARIANTS.get(bm, {})
        if not g1_descs:
            continue

        hdr = f"{'tokens':>8} |"
        for v in sorted(g1_descs.keys()):
            hdr += f" g1v{v:>1}_us |"
        hdr += " best | desc"
        print(hdr)
        print("-" * len(hdr))

        for token_num in token_counts:
            data = setup_moe_data(token_num)
            line = f"{token_num:>8} |"
            best_us = float("inf")
            best_v = -1

            for v in sorted(g1_descs.keys()):
                set_variant(v, 0)
                try:
                    get_2stage_cfgs.cache_clear()
                    r = bench_fused_moe(data, bm, args.warmup, args.iters)
                    total_us = r["total_us"]
                    line += f" {total_us:>7.1f} |"
                    key = f"bm{bm}_t{token_num}_g1v{v}"
                    all_results[key] = total_us
                    if total_us < best_us:
                        best_us = total_us
                        best_v = v
                except Exception as e:
                    line += f" {'FAIL':>7} |"
                    print(f"  [WARN] bm={bm}, t={token_num}, g1v={v}: {e}")

            line += f"  v{best_v} | {g1_descs.get(best_v, '?')}"
            print(line)

        print(f"\n--- block_m={bm}: Gemm2 variants (G1=default) ---")
        g2_descs = G2_VARIANTS.get(bm, {})
        if not g2_descs:
            continue

        hdr = f"{'tokens':>8} |"
        for v in sorted(g2_descs.keys()):
            hdr += f" g2v{v:>1}_us |"
        hdr += " best | desc"
        print(hdr)
        print("-" * len(hdr))

        for token_num in token_counts:
            data = setup_moe_data(token_num)
            line = f"{token_num:>8} |"
            best_us = float("inf")
            best_v = -1

            for v in sorted(g2_descs.keys()):
                set_variant(0, v)
                try:
                    get_2stage_cfgs.cache_clear()
                    r = bench_fused_moe(data, bm, args.warmup, args.iters)
                    total_us = r["total_us"]
                    line += f" {total_us:>7.1f} |"
                    key = f"bm{bm}_t{token_num}_g2v{v}"
                    all_results[key] = total_us
                    if total_us < best_us:
                        best_us = total_us
                        best_v = v
                except Exception as e:
                    line += f" {'FAIL':>7} |"
                    print(f"  [WARN] bm={bm}, t={token_num}, g2v={v}: {e}")

            line += f"  v{best_v} | {g2_descs.get(best_v, '?')}"
            print(line)

    set_variant(0, 0)
    return all_results


def phase3_combined_best(token_counts, phase2_results, args):
    """Phase 3: Test best g1+g2 variant combinations."""
    print()
    print("=" * 70)
    print("Phase 3: Combined best variants")
    print("=" * 70)

    for bm in [16, 32]:
        # Find best g1 and g2 variants per token count
        print(f"\n--- block_m={bm}: Best combinations ---")
        hdr = f"{'tokens':>8} | default_us | best_g1v | best_g2v | combined_us | speedup"
        print(hdr)
        print("-" * len(hdr))

        for token_num in token_counts:
            # Find best g1 variant
            best_g1 = 0
            best_g1_us = phase2_results.get(f"bm{bm}_t{token_num}_g1v0", float("inf"))
            for v in range(4):
                us = phase2_results.get(f"bm{bm}_t{token_num}_g1v{v}", float("inf"))
                if us < best_g1_us:
                    best_g1_us = us
                    best_g1 = v

            # Find best g2 variant
            best_g2 = 0
            best_g2_us = phase2_results.get(f"bm{bm}_t{token_num}_g2v0", float("inf"))
            for v in range(4):
                us = phase2_results.get(f"bm{bm}_t{token_num}_g2v{v}", float("inf"))
                if us < best_g2_us:
                    best_g2_us = us
                    best_g2 = v

            # Get default
            default_us = phase2_results.get(f"bm{bm}_t{token_num}_g1v0", 0)
            if default_us == 0:
                continue

            # Test combined
            set_variant(best_g1, best_g2)
            data = setup_moe_data(token_num)
            try:
                get_2stage_cfgs.cache_clear()
                r = bench_fused_moe(data, bm, args.warmup, args.iters)
                combined_us = r["total_us"]
                speedup = default_us / combined_us if combined_us > 0 else 0
                print(
                    f"{token_num:>8} | {default_us:>10.1f} | "
                    f"g1v{best_g1} ({G1_VARIANTS.get(bm, {}).get(best_g1, '?')[:15]:>15}) | "
                    f"g2v{best_g2} ({G2_VARIANTS.get(bm, {}).get(best_g2, '?')[:15]:>15}) | "
                    f"{combined_us:>11.1f} | {speedup:.3f}x"
                )
            except Exception as e:
                print(f"  [WARN] bm={bm}, t={token_num}, g1={best_g1}, g2={best_g2}: {e}")

    set_variant(0, 0)


def main():
    parser = argparse.ArgumentParser(description="Sweep MoE kernel tile variants")
    parser.add_argument(
        "--tokens", nargs="+", type=int,
        default=[1, 4, 16, 64, 128, 256],
        help="Token counts (kept < 512 for a16w4 path)",
    )
    parser.add_argument(
        "--block-m", nargs="+", type=int, default=[16, 32, 48, 64, 128],
        help="block_m values for Phase 1",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--phase", nargs="+", type=int, default=[1, 2, 3],
        help="Which phases to run (1=block_m sweep, 2=variant sweep, 3=combined)",
    )
    args = parser.parse_args()

    # Clamp tokens to < 512 (a16w4 boundary)
    args.tokens = [t for t in args.tokens if t < 512]

    print(f"GPT-OSS MoE Tile Variant Sweep")
    print(f"  model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, experts={EXPERTS}, topk={TOPK}")
    print(f"  activation=Swiglu, quant=per_1x32 (MXFP4)")
    print(f"  tokens: {args.tokens}")
    print(f"  phases: {args.phase}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print()

    # JIT warmup
    jit_warmup(args.block_m, G1_VARIANTS, G2_VARIANTS)

    # Phase 1: block_m sweep
    if 1 in args.phase:
        phase1_block_m_sweep(args.tokens, args.block_m, args)

    # Phase 2: tile variant sweep
    phase2_results = {}
    if 2 in args.phase:
        phase2_results = phase2_variant_sweep(args.tokens, args)

    # Phase 3: combined best
    if 3 in args.phase and phase2_results:
        phase3_combined_best(args.tokens, phase2_results, args)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
