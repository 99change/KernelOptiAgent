#!/usr/bin/env python3
"""
run_experiments.py
KernelBench Level-1 Style Experiment Runner
============================================
Runs two systems on every benchmark kernel and records speedup results.

Systems compared
----------------
1. KernelAgent  — our full pipeline (Profiler → Analyzer → Optimizer)
2. E2E Baseline — bare LLM optimization (identical to baseline_e2e.py logic)

Usage
-----
    # full run on all benchmarks (slow: ~10 min/kernel × 2 systems)
    python run_experiments.py

    # run a specific subset
    python run_experiments.py --kernels relu softmax

    # quick smoke test with 1 LLM try
    python run_experiments.py --baseline-tries 1 --rounds 1

    # dry-run to verify file discovery
    python run_experiments.py --dry-run

Results are saved to:
    results/experiment_YYYYMMDD_HHMMSS.csv    (machine-readable)
    results/experiment_YYYYMMDD_HHMMSS.txt    (human-readable summary)
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# ── make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from core.config import DASHSCOPE_BASE_URL, LLM_CONFIG, LLMConfig
from tools.kernel_tools import compile_and_test

# ── benchmark catalogue ────────────────────────────────────────────────────────
BENCHMARKS = {
    "matmul": {
        "file"       : "examples/matmul_naive.cu",
        "description": "GEMM 1024×1024 fp32 (KernelBench L1: linear algebra)",
    },
    "vector_add": {
        "file"       : "examples/vector_add.cu",
        "description": "Vector addition 16 M elements (KernelBench L1: elementwise binary)",
    },
    "relu": {
        "file"       : "benchmarks/relu_naive.cu",
        "description": "ReLU 33.5 M elements (KernelBench L1: elementwise unary)",
    },
    "softmax": {
        "file"       : "benchmarks/softmax_naive.cu",
        "description": "Row-wise softmax 4096×1024 (KernelBench L1: reduction)",
    },
    "layernorm": {
        "file"       : "benchmarks/layernorm_naive.cu",
        "description": "Layer norm 4096×1024 (KernelBench L1: compound reduction)",
    },
    "transpose": {
        "file"       : "benchmarks/transpose_naive.cu",
        "description": "4096×4096 transpose (KernelBench L1: memory pattern)",
    },
}

# ── E2E baseline LLM prompt ────────────────────────────────────────────────────
E2E_PROMPT = """\
You are a CUDA optimization expert. Below is a CUDA kernel implementation.
Your task: rewrite it to be as fast as possible on a modern NVIDIA GPU (sm_120).

Rules:
- Output ONLY the complete, compilable CUDA C++ source file. No markdown, no explanation.
- Keep the same function signature and behavior (same inputs/outputs, same correctness).
- Use any optimization you judge appropriate: vectorized loads, shared memory, \
warp intrinsics, loop unrolling, occupancy tuning, etc.

Original kernel:
{code}
"""


def _strip_markdown(text: str) -> str:
    """Remove ```...``` fences that LLMs sometimes add."""
    if "```" not in text:
        return text.strip()
    lines, inside, result = text.splitlines(), False, []
    for line in lines:
        if line.strip().startswith("```"):
            inside = not inside
            continue
        if inside:
            result.append(line)
    return "\n".join(result).strip()


def run_e2e_baseline(client: OpenAI, model: str, original_code: str,
                     tries: int, baseline_ms: float) -> dict:
    """Run the bare-LLM baseline (same logic as baseline_e2e.py)."""
    best_ms = baseline_ms
    best_speedup = 0.0
    success_tries = 0

    for i in range(tries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.7,
                max_tokens=4096,
                messages=[{"role": "user", "content": E2E_PROMPT.format(code=original_code)}],
            )
            optimized = _strip_markdown(resp.choices[0].message.content or "")
            if not optimized:
                continue
            result = compile_and_test(optimized)
            if not result.success:
                continue
            success_tries += 1
            if result.exec_time_ms < best_ms:
                best_ms = result.exec_time_ms
                best_speedup = (baseline_ms - best_ms) / baseline_ms * 100
        except Exception:
            pass

    return {
        "best_ms"    : best_ms,
        "speedup_pct": best_speedup,
        "success"    : success_tries > 0,
        "tries_ok"   : success_tries,
    }


def run_kernelagent(kernel_file: str, kernel_code: str, baseline_ms: float,
                    rounds: int, best_of_n: int, model: str) -> dict:
    """Run KernelAgent pipeline via main.run()."""
    from main import run as agent_run
    from core.config import LLMConfig

    cfg = LLMConfig(model=model)
    t0 = time.time()
    try:
        report = agent_run(
            kernel_code=kernel_code,
            mock=False,
            max_rounds=rounds,
            llm_config=cfg,
            best_of_n=best_of_n,
        )
        elapsed = time.time() - t0

        # use report's top-level optimized_time_ms
        best_ms = getattr(report, "optimized_time_ms", baseline_ms) or baseline_ms
        speedup = (baseline_ms - best_ms) / baseline_ms * 100
        return {
            "best_ms"    : best_ms,
            "speedup_pct": speedup,
            "success"    : True,
            "elapsed_s"  : elapsed,
            "rounds_done": 0,
            "error"      : "",
        }
    except Exception as e:
        return {
            "best_ms"    : baseline_ms,
            "speedup_pct": 0.0,
            "success"    : False,
            "elapsed_s"  : time.time() - t0,
            "rounds_done": 0,
            "error"      : str(e)[:200],
        }


def print_table(rows: list[dict], file=sys.stdout):
    """Pretty-print results table."""
    cols = ["kernel", "baseline_ms", "agent_ms", "agent_%", "e2e_ms", "e2e_%"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in cols}

    def fmt(r, c):
        v = r.get(c, "")
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v) if v is not None else "—"

    def sep():
        return "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"

    def row_str(r):
        return "|" + "|".join(f" {fmt(r,c):<{widths[c]}} " for c in cols) + "|"

    print(sep(), file=file)
    print("|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|", file=file)
    print(sep(), file=file)
    for r in rows:
        print(row_str(r), file=file)
    print(sep(), file=file)


def main():
    parser = argparse.ArgumentParser(description="KernelBench Level-1 experiment runner")
    parser.add_argument("--kernels", nargs="*", choices=list(BENCHMARKS), default=None,
                        help="subset of kernels to run (default: all)")
    parser.add_argument("--model", default="qwen3.5-flash-2026-02-23",
                        help="LLM model name")
    parser.add_argument("--rounds", type=int, default=3,
                        help="KernelAgent max optimization rounds")
    parser.add_argument("--best-of-n", type=int, default=1, dest="best_of_n",
                        help="KernelAgent Best-of-N candidates per strategy")
    parser.add_argument("--baseline-tries", type=int, default=3, dest="baseline_tries",
                        help="E2E baseline LLM tries")
    parser.add_argument("--skip-agent", action="store_true",
                        help="Skip KernelAgent, only run E2E baseline")
    parser.add_argument("--skip-e2e", action="store_true",
                        help="Skip E2E baseline, only run KernelAgent")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print kernel list and exit")
    args = parser.parse_args()

    selected = args.kernels or list(BENCHMARKS)

    # ── validate files ────────────────────────────────────────────────────────
    missing = []
    for k in selected:
        p = PROJECT_ROOT / BENCHMARKS[k]["file"]
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: Missing benchmark files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("Dry-run mode. Kernels that would be evaluated:")
        for k in selected:
            print(f"  [{k:12s}] {BENCHMARKS[k]['file']}")
            print(f"             {BENCHMARKS[k]['description']}")
        return

    # ── LLM client ───────────────────────────────────────────────────────────
    cfg = LLMConfig(model=args.model)
    if not cfg.api_key:
        print("ERROR: DASHSCOPE_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=cfg.api_key, base_url=DASHSCOPE_BASE_URL, timeout=180.0)

    # ── results ───────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"experiment_{ts}.csv"
    txt_path = results_dir / f"experiment_{ts}.txt"

    table_rows = []
    csv_rows = []

    total = len(selected)
    for idx, kernel_name in enumerate(selected, 1):
        info = BENCHMARKS[kernel_name]
        kernel_path = PROJECT_ROOT / info["file"]
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] {kernel_name.upper()} — {info['description']}")
        print(f"{'='*70}")

        with open(kernel_path) as f:
            kernel_code = f.read()

        # baseline compile
        print("  Compiling baseline...")
        bl = compile_and_test(kernel_code)
        if not bl.success:
            print(f"  ERROR: baseline compile failed: {bl.error[:100]}")
            row = {"kernel": kernel_name, "baseline_ms": "FAIL",
                   "agent_ms": "—", "agent_%": "—",
                   "e2e_ms": "—", "e2e_%": "—"}
            table_rows.append(row)
            continue
        print(f"  Baseline: {bl.exec_time_ms:.3f} ms")

        agent_result = {"best_ms": bl.exec_time_ms, "speedup_pct": 0.0,
                        "success": False, "rounds_done": 0, "elapsed_s": 0}
        e2e_result   = {"best_ms": bl.exec_time_ms, "speedup_pct": 0.0,
                        "success": False, "tries_ok": 0}

        # ── KernelAgent ───────────────────────────────────────────────────────
        if not args.skip_agent:
            print(f"\n  >>> KernelAgent (rounds={args.rounds}, N={args.best_of_n})")
            agent_result = run_kernelagent(
                kernel_file=str(kernel_path),
                kernel_code=kernel_code,
                baseline_ms=bl.exec_time_ms,
                rounds=args.rounds,
                best_of_n=args.best_of_n,
                model=args.model,
            )
            status = f"{agent_result['speedup_pct']:+.1f}%" if agent_result["success"] else f"FAIL: {agent_result['error']}"
            print(f"  KernelAgent result: {agent_result['best_ms']:.3f} ms  ({status})")

        # ── E2E Baseline ──────────────────────────────────────────────────────
        if not args.skip_e2e:
            print(f"\n  >>> E2E Baseline (tries={args.baseline_tries})")
            e2e_result = run_e2e_baseline(
                client=client,
                model=args.model,
                original_code=kernel_code,
                tries=args.baseline_tries,
                baseline_ms=bl.exec_time_ms,
            )
            status = f"{e2e_result['speedup_pct']:+.1f}%" if e2e_result["success"] else "FAIL"
            print(f"  E2E Baseline result: {e2e_result['best_ms']:.3f} ms  ({status})")

        row = {
            "kernel"      : kernel_name,
            "baseline_ms" : f"{bl.exec_time_ms:.3f}",
            "agent_ms"    : f"{agent_result['best_ms']:.3f}",
            "agent_%"     : f"{agent_result['speedup_pct']:+.1f}",
            "e2e_ms"      : f"{e2e_result['best_ms']:.3f}",
            "e2e_%"       : f"{e2e_result['speedup_pct']:+.1f}",
        }
        table_rows.append(row)

        csv_rows.append({
            "kernel"              : kernel_name,
            "description"         : info["description"],
            "baseline_ms"         : bl.exec_time_ms,
            "agent_best_ms"       : agent_result["best_ms"],
            "agent_speedup_pct"   : agent_result["speedup_pct"],
            "agent_success"       : agent_result["success"],
            "agent_rounds"        : agent_result.get("rounds_done", ""),
            "agent_elapsed_s"     : agent_result.get("elapsed_s", ""),
            "e2e_best_ms"         : e2e_result["best_ms"],
            "e2e_speedup_pct"     : e2e_result["speedup_pct"],
            "e2e_success"         : e2e_result["success"],
            "e2e_tries_ok"        : e2e_result.get("tries_ok", ""),
            "model"               : args.model,
            "agent_rounds_cfg"    : args.rounds,
            "agent_best_of_n_cfg" : args.best_of_n,
        })

    # ── write CSV ─────────────────────────────────────────────────────────────
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # ── print & save summary ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT SUMMARY  [{ts}]")
    print(f"Model: {args.model}   Agent-rounds: {args.rounds}   Best-of-N: {args.best_of_n}")
    print(f"{'='*70}\n")
    print_table(table_rows)

    # compute aggregate stats
    agent_speedups = [float(r["agent_%"]) for r in table_rows
                      if r["agent_%"] not in ("—", "FAIL")]
    e2e_speedups   = [float(r["e2e_%"]) for r in table_rows
                      if r["e2e_%"] not in ("—", "FAIL")]

    if agent_speedups:
        print(f"\n  KernelAgent avg speedup : {sum(agent_speedups)/len(agent_speedups):+.1f}%")
    if e2e_speedups:
        print(f"  E2E Baseline avg speedup: {sum(e2e_speedups)/len(e2e_speedups):+.1f}%")

    summary_text = []
    summary_text.append(f"EXPERIMENT SUMMARY  [{ts}]")
    summary_text.append(f"Model: {args.model}   Rounds: {args.rounds}   N: {args.best_of_n}\n")
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_text))
        # rebuild table to file
        print_table(table_rows, file=f)
        if agent_speedups:
            f.write(f"\nKernelAgent avg speedup : {sum(agent_speedups)/len(agent_speedups):+.1f}%\n")
        if e2e_speedups:
            f.write(f"E2E Baseline avg speedup: {sum(e2e_speedups)/len(e2e_speedups):+.1f}%\n")

    print(f"\nCSV saved: {csv_path}")
    print(f"TXT saved: {txt_path}")


if __name__ == "__main__":
    main()
