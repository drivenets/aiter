#!/usr/bin/env python3
"""
Benchmark: 16x16 vs 32x32 warp tile MXFP4 MoE kernels for GPT-OSS.

Compares the existing 16x16 kernel (K_align=256, more padding) against
the new 32x32 kernel (K_align=128, less padding) on GPT-OSS MoE shapes.

GPT-OSS per-GPU MoE dimensions:
  hidden_size = 2880, intermediate_size/tp = 360
  16x16: hidden→3072, inter→512 (42% waste on inter)
  32x32: hidden→2944, inter→384 (6.7% waste on inter)

Usage:
    # Must set AITER_MOE_WARP32=0 or =1 BEFORE running (JIT compiles at first call)
    AITER_MOE_WARP32=0 python bench_32x32_moe.py   # 16x16 baseline
    AITER_MOE_WARP32=1 python bench_32x32_moe.py   # 32x32 new kernel

    # Or run both:
    ./bench_32x32_moe.py --both
"""

import torch
import argparse
import sys
import os
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

# GPT-OSS MoE raw dimensions (before padding)
RAW_HIDDEN = 2880
RAW_INTER_PER_TP = 360  # 2880 / 8
EXPERTS = 128
TOPK = 4
DTYPE = dtypes.bf16
ACTIVATION = ActivationType.Swiglu


def round_up(x, a):
    return ((x + a - 1) // a) * a


def get_config():
    """Return padded dimensions and NLane based on AITER_MOE_WARP32 env var."""
    use_warp32 = os.environ.get("AITER_MOE_WARP32", "0") != "0"
    k_align = 128 if use_warp32 else 256
    n_lane = 32 if use_warp32 else 16

    hidden = round_up(RAW_HIDDEN, k_align)
    inter = round_up(RAW_INTER_PER_TP, k_align)

    return {
        "warp": "32x32" if use_warp32 else "16x16",
        "k_align": k_align,
        "n_lane": n_lane,
        "hidden": hidden,
        "inter": inter,
        "hidden_pad": hidden - RAW_HIDDEN,
        "inter_pad": inter - RAW_INTER_PER_TP,
    }


def setup_data(token_num, block_m, cfg):
    """Prepare pre-sorted, pre-quantized data for raw kernel calls."""
    torch.manual_seed(42)
    hidden = cfg["hidden"]
    inter = cfg["inter"]
    n_lane = cfg["n_lane"]

    hidden_states = torch.randn((token_num, hidden), dtype=DTYPE)

    # w1: gate+up fused [E, inter*2, hidden], w2: down [E, hidden, inter]
    w1 = torch.randn((EXPERTS, inter * 2, hidden), dtype=DTYPE) / 10
    w2 = torch.randn((EXPERTS, hidden, inter), dtype=DTYPE) / 10

    # Quantize to MXFP4
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Shuffle for CK-tile a16w4
    w1_qt = shuffle_weight_a16w4(w1_qt, n_lane, True)
    w1_scale = shuffle_scale_a16w4(w1_scale, EXPERTS, True, n_lane=n_lane)
    w2_qt = shuffle_weight_a16w4(w2_qt, n_lane, False)
    w2_scale = shuffle_scale_a16w4(w2_scale, EXPERTS, False, n_lane=n_lane)

    # Routing
    score = torch.randn((token_num, EXPERTS), dtype=DTYPE)
    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)

    # Moe sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, EXPERTS, hidden, DTYPE, block_m,
    )

    a1 = hidden_states.to(DTYPE)

    _, n1, k1 = w1_qt.shape
    _, k2, n2 = w2_qt.shape
    D = n2 * 2  # fp4x2 packing

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
        "token_num": token_num,
        "D": D,
        "hidden": hidden,
        "inter": inter,
    }


def bench_stage1(data, block_m, num_warmup=10, num_iters=50):
    token_num = data["token_num"]
    w1 = data["w1"]
    D = data["D"]

    for _ in range(num_warmup):
        out = torch.empty((token_num, TOPK, D), dtype=DTYPE, device="cuda")
        aiter.moe_cktile2stages_gemm1(
            data["a1"], w1, out,
            data["sorted_ids"], data["sorted_expert_ids"], data["num_valid_ids"],
            TOPK, 0, 0, None, None,
            data["w1_scale"], None, ACTIVATION, block_m, 1,
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
            TOPK, 0, 0, None, None,
            data["w1_scale"], None, ACTIVATION, block_m, 1,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times_us = sorted(times_us[2:])
    median_us = times_us[len(times_us) // 2]

    m_eff = token_num * TOPK
    hidden = data["hidden"]
    inter = data["inter"]
    flops = m_eff * (inter * 2) * hidden * 2
    tflops = flops / median_us / 1e6

    return {"us": median_us, "tflops": tflops}


def bench_stage2(data, block_m, num_warmup=10, num_iters=50):
    token_num = data["token_num"]
    inter = data["inter"]
    hidden = data["hidden"]

    a2 = torch.randn((token_num, TOPK, inter), dtype=DTYPE, device="cuda")
    moe_out = torch.empty((token_num, hidden), dtype=DTYPE, device="cuda")

    for _ in range(num_warmup):
        aiter.moe_cktile2stages_gemm2(
            a2, data["w2"], moe_out,
            data["sorted_ids"], data["sorted_expert_ids"], data["num_valid_ids"],
            TOPK, 0, 0,
            data["sorted_weights"], None,
            data["w2_scale"], None, ACTIVATION, block_m,
        )
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        aiter.moe_cktile2stages_gemm2(
            a2, data["w2"], moe_out,
            data["sorted_ids"], data["sorted_expert_ids"], data["num_valid_ids"],
            TOPK, 0, 0,
            data["sorted_weights"], None,
            data["w2_scale"], None, ACTIVATION, block_m,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times_us = sorted(times_us[2:])
    median_us = times_us[len(times_us) // 2]

    m_eff = token_num * TOPK
    flops = m_eff * hidden * inter * 2
    tflops = flops / median_us / 1e6

    return {"us": median_us, "tflops": tflops}


def run_single(args):
    """Run benchmark for current AITER_MOE_WARP32 setting."""
    cfg = get_config()

    print(f"\n{'='*70}")
    print(f"WARP TILE: {cfg['warp']}  (K_align={cfg['k_align']}, NLane={cfg['n_lane']})")
    print(f"  hidden: {RAW_HIDDEN} → {cfg['hidden']} (+{cfg['hidden_pad']})")
    print(f"  inter:  {RAW_INTER_PER_TP} → {cfg['inter']} (+{cfg['inter_pad']})")
    print(f"  gemm1: M=tok*{TOPK}, N={cfg['inter']*2}, K={cfg['hidden']}")
    print(f"  gemm2: M=tok*{TOPK}, N={cfg['hidden']}, K={cfg['inter']}")
    print(f"{'='*70}")

    # JIT warmup
    print("Triggering JIT compilation (may take minutes on first run)...")
    safe_data = setup_data(64, 32, cfg)
    bench_stage1(safe_data, 32, num_warmup=2, num_iters=3)
    bench_stage2(safe_data, 32, num_warmup=2, num_iters=3)
    print("  JIT done.\n")

    results = {}
    for bm in args.block_m:
        for token_num in args.tokens:
            data = setup_data(token_num, bm, cfg)
            m_eff = token_num * TOPK

            r1 = bench_stage1(data, bm, args.warmup, args.iters)
            r2 = bench_stage2(data, bm, args.warmup, args.iters)

            key = f"t{token_num}_bm{bm}"
            results[key] = {
                "tokens": token_num,
                "block_m": bm,
                "m_eff": m_eff,
                "gemm1_us": r1["us"],
                "gemm1_tflops": r1["tflops"],
                "gemm2_us": r2["us"],
                "gemm2_tflops": r2["tflops"],
                "total_us": r1["us"] + r2["us"],
            }

            print(f"  tok={token_num:>4} bm={bm:>2} | "
                  f"gemm1: {r1['us']:>7.1f}us ({r1['tflops']:>5.1f}TF) | "
                  f"gemm2: {r2['us']:>7.1f}us ({r2['tflops']:>5.1f}TF) | "
                  f"total: {r1['us']+r2['us']:>7.1f}us")

    return {"config": cfg, "results": results}


def run_both(args):
    """Run both 16x16 and 32x32 in separate processes, then compare."""
    script = os.path.abspath(__file__)
    results = {}

    for warp32 in ["0", "1"]:
        label = "32x32" if warp32 == "1" else "16x16"
        print(f"\n{'#'*70}")
        print(f"# Running {label} kernel...")
        print(f"{'#'*70}")

        env = os.environ.copy()
        env["AITER_MOE_WARP32"] = warp32

        cmd = [
            sys.executable, script, "--single",
            "--tokens", *[str(t) for t in args.tokens],
            "--block-m", *[str(b) for b in args.block_m],
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
        ]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-500:])

        # Parse JSON results from last line
        for line in reversed(result.stdout.strip().split("\n")):
            if line.startswith("JSON:"):
                results[label] = json.loads(line[5:])
                break
        else:
            print(f"WARNING: No JSON output from {label} run")
            results[label] = None

    # Compare
    if results.get("16x16") and results.get("32x32"):
        print(f"\n{'='*70}")
        print("COMPARISON: 16x16 vs 32x32 warp tile")
        print(f"{'='*70}")

        r16 = results["16x16"]["results"]
        r32 = results["32x32"]["results"]

        hdr = f"{'key':>12} | {'16x16 total':>12} | {'32x32 total':>12} | {'speedup':>8} | {'32x32 FLOP%':>11}"
        print(hdr)
        print("-" * len(hdr))

        for key in sorted(r16.keys()):
            if key in r32:
                t16 = r16[key]["total_us"]
                t32 = r32[key]["total_us"]
                speedup = t16 / t32 if t32 > 0 else float("inf")
                # FLOP reduction
                flop_ratio = (
                    (r32[key]["m_eff"] * (results["32x32"]["config"]["inter"] * 2 * results["32x32"]["config"]["hidden"] +
                     results["32x32"]["config"]["hidden"] * results["32x32"]["config"]["inter"])) /
                    (r16[key]["m_eff"] * (results["16x16"]["config"]["inter"] * 2 * results["16x16"]["config"]["hidden"] +
                     results["16x16"]["config"]["hidden"] * results["16x16"]["config"]["inter"]))
                ) * 100
                print(f"{key:>12} | {t16:>10.1f}us | {t32:>10.1f}us | {speedup:>7.2f}x | {flop_ratio:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="16x16 vs 32x32 MoE kernel benchmark")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--block-m", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--both", action="store_true", help="Run both 16x16 and 32x32 in subprocesses")
    parser.add_argument("--single", action="store_true", help="Run single config (used internally)")
    args = parser.parse_args()

    args.tokens = [t for t in args.tokens if t < 512]

    if args.both:
        run_both(args)
    else:
        result = run_single(args)
        # Print JSON for subprocess parsing
        print(f"JSON:{json.dumps(result)}")


if __name__ == "__main__":
    main()
