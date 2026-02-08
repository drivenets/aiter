#!/usr/bin/env python3
"""
Micro-benchmark for CK-tile MoE (a16w4) kernels with GPT-OSS shapes.

Tests different block_m values (16, 32, 64, 128) across various token counts
to find optimal configurations for GPT-OSS (model_dim=3072, inter_dim=512,
experts=128, topk=4, activation=Swiglu, MXFP4 weights).

Usage:
    python bench_gptoss_moe.py [--block-m 16 32 64 128] [--tokens 1 4 16 ...]
"""

import torch
import time
import argparse
import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    get_block_size_M,
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


def setup_moe_data(token_num):
    """Create test data matching GPT-OSS MoE shapes with MXFP4 weights."""
    torch.manual_seed(42)

    hidden_states = torch.randn((token_num, MODEL_DIM), dtype=DTYPE)
    # g1u1: w1 has gate+up = 2*inter_dim
    w1 = torch.randn((EXPERTS, INTER_DIM * 2, MODEL_DIM), dtype=DTYPE) / 10
    w2 = torch.randn((EXPERTS, MODEL_DIM, INTER_DIM), dtype=DTYPE) / 10

    # Quantize weights to MXFP4 (per_1x32)
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    score = torch.randn((token_num, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    # Shuffle for CK-tile a16w4 path
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


def bench_fused_moe(
    hidden_states, w1_qt, w2_qt, w1_scale, w2_scale, topk_weights, topk_ids,
    block_m, num_warmup=5, num_iters=20,
):
    """Benchmark fused_moe with explicit block_size_M override."""
    token_num = hidden_states.shape[0]

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
    # Drop first 2 and take median of the rest
    if len(times_ms) > 4:
        times_ms = sorted(times_ms[2:])
        median_ms = times_ms[len(times_ms) // 2]
    else:
        median_ms = sorted(times_ms)[len(times_ms) // 2]

    total_us = median_ms * 1000

    # Compute TFLOPS (approximate)
    m_eff = token_num * TOPK
    flop_g1 = m_eff * (INTER_DIM * 2) * MODEL_DIM * 2
    flop_g2 = m_eff * MODEL_DIM * INTER_DIM * 2
    tflops = (flop_g1 + flop_g2) / total_us / 1e6

    return {"total_us": total_us, "tflops": tflops}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CK-tile MoE kernels for GPT-OSS shapes"
    )
    parser.add_argument(
        "--block-m", nargs="+", type=int, default=[16, 32, 64, 128],
        help="block_m values to test",
    )
    parser.add_argument(
        "--tokens", nargs="+", type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help="Token counts to test",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    print(f"GPT-OSS MoE Benchmark (a16w4 CK-tile path)")
    print(f"  model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, experts={EXPERTS}, topk={TOPK}")
    print(f"  activation=Swiglu, quant=per_1x32 (MXFP4)")
    print(f"  block_m candidates: {args.block_m}")
    print(f"  token counts: {args.tokens}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print()

    # Trigger JIT compilation with a safe token count
    print("Triggering JIT compilation (may take several minutes)...")
    valid_block_ms = []
    for bm in args.block_m:
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
            valid_block_ms.append(bm)
            print(f"  block_m={bm}: OK")
        except Exception as e:
            print(f"  block_m={bm}: FAILED - {e}")
    print()

    if not valid_block_ms:
        print("ERROR: No valid block_m values. Exiting.")
        return

    # Clear LRU caches that might interfere
    from aiter.fused_moe import get_2stage_cfgs
    get_2stage_cfgs.cache_clear()

    # Header
    hdr = f"{'tokens':>8} | {'heur_bm':>7} |"
    for bm in valid_block_ms:
        hdr += f"  bm={bm:>3}_us |  bm={bm:>3}_tf |"
    hdr += " best_bm | speedup"
    print(hdr)
    print("-" * len(hdr))

    results = []

    for token_num in args.tokens:
        heuristic_bm = get_block_size_M(token_num, TOPK, EXPERTS, INTER_DIM)
        data = setup_moe_data(token_num)
        row = {"tokens": token_num, "heuristic_bm": heuristic_bm}

        line = f"{token_num:>8} | bm={heuristic_bm:>3} |"
        best_us = float("inf")
        best_bm = -1
        heuristic_us = None

        for bm in valid_block_ms:
            try:
                # Clear cache to avoid stale metadata
                get_2stage_cfgs.cache_clear()
                r = bench_fused_moe(
                    *data, block_m=bm,
                    num_warmup=args.warmup, num_iters=args.iters,
                )
                total_us = r["total_us"]
                tflops = r["tflops"]
                row[f"bm{bm}_us"] = total_us
                row[f"bm{bm}_tflops"] = tflops
                line += f" {total_us:>9.1f} | {tflops:>8.2f} |"

                if total_us < best_us:
                    best_us = total_us
                    best_bm = bm
                if bm == heuristic_bm:
                    heuristic_us = total_us
            except Exception as e:
                line += f" {'FAIL':>9} | {'N/A':>8} |"
                row[f"bm{bm}_us"] = None
                print(f"  [WARN] tokens={token_num}, bm={bm}: {e}")

        speedup = heuristic_us / best_us if heuristic_us and best_us < float("inf") else 0
        line += f" bm={best_bm:>3} | {speedup:>5.2f}x"
        row["best_bm"] = best_bm
        row["best_us"] = best_us
        row["speedup"] = speedup

        print(line)
        results.append(row)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY: Optimal block_m for GPT-OSS MoE shapes")
    print("=" * 80)
    for r in results:
        marker = " *** IMPROVEMENT" if r.get("speedup", 0) > 1.02 else ""
        print(
            f"  tokens={r['tokens']:>6}: best=bm{r['best_bm']:>3} "
            f"({r['best_us']:>8.1f} us), "
            f"heuristic=bm{r['heuristic_bm']:>3}, "
            f"speedup={r.get('speedup', 0):.2f}x{marker}"
        )


if __name__ == "__main__":
    main()
