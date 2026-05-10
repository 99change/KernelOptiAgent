#!/usr/bin/env python3
"""
fill_results.py
从最新的实验 CSV 中读取结果，自动替换 sections/experiments.tex 中的 \RESULTVAL{} 占位符。

用法：
    python fill_results.py                   # 自动选最新 CSV
    python fill_results.py results/foo.csv   # 指定 CSV
"""

import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TEX_FILE = PROJECT_ROOT / "sections" / "experiments.tex"

# 已知的静态数据（不来自 CSV）
STATIC = {
    "gemm_baseline"  : "91.87",
    "gemm_agent"     : r"\textbf{+32.5}",
    "gemm_e2e"       : "+27.4",
    "relu_baseline"  : "0.456",
    "relu_e2e"       : "+0.0",
    "softmax_baseline": "0.460",
    "softmax_e2e"    : "+0.0",
    "layernorm_baseline": "0.448",
    "layernorm_e2e"  : r"\textbf{+87.9}",
    "transpose_baseline": "0.428",
    "transpose_e2e"  : "+22.8",
    "vecadd_baseline": "0.627",
    "vecadd_e2e"     : "+0.0",
    "matmul_e2e_ms"  : "65.30",
    "matmul_e2e_pct" : "+27.4",
}


def load_latest_csv() -> Path | None:
    csvs = sorted(PROJECT_ROOT.glob("results/experiment_*.csv"), key=lambda p: p.stat().st_mtime)
    return csvs[-1] if csvs else None


def load_csv(path: Path) -> dict[str, str]:
    """Return a dict keyed by 'kernel' with all row data."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["kernel"]] = row
    return rows


def compute_values(csv_rows: dict) -> dict[str, str]:
    """Map RESULTVAL keys to replacement strings."""
    vals = dict(STATIC)

    kernel_map = {
        "softmax"    : ("softmax_agent",),
        "layernorm"  : ("layernorm_agent",),
        "transpose"  : ("transpose_agent",),
        "relu"       : ("relu_agent",),
        "vector_add" : ("vecadd_agent",),
        "matmul"     : (),
    }

    all_agent_speedups = []

    for kernel, data in csv_rows.items():
        spd = data.get("agent_speedup_pct", "")
        ok  = data.get("agent_success", "False").lower() == "true"
        ms  = data.get("agent_best_ms", "")

        if kernel in kernel_map and kernel_map[kernel]:
            key = kernel_map[kernel][0]
            if ok and spd:
                try:
                    spd_f = float(spd)
                    all_agent_speedups.append(spd_f)
                    vals[key] = f"{spd_f:+.1f}"
                except ValueError:
                    vals[key] = "—"
            else:
                vals[key] = "—"

    # also count GEMM agent result
    all_agent_speedups.append(32.5)   # from demo run

    if all_agent_speedups:
        avg = sum(all_agent_speedups) / len(all_agent_speedups)
        vals["agent_avg"] = f"{avg:+.1f}"

    return vals


def replace_placeholders(tex: str, vals: dict) -> tuple[str, list[str]]:
    replaced = []
    pattern = re.compile(r"\\RESULTVAL\{([^}]+)\}")

    def repl(m):
        key = m.group(1)
        if key in vals:
            replaced.append(key)
            return vals[key]
        return m.group(0)   # leave unchanged

    return pattern.sub(repl, tex), replaced


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else load_latest_csv()
    if not csv_path or not csv_path.exists():
        print("ERROR: no experiment CSV found in results/")
        sys.exit(1)
    print(f"Using CSV: {csv_path}")

    csv_rows = load_csv(csv_path)
    vals = compute_values(csv_rows)

    tex = TEX_FILE.read_text()
    new_tex, replaced = replace_placeholders(tex, vals)

    TEX_FILE.write_text(new_tex)
    print(f"Replaced {len(replaced)} placeholder(s): {replaced}")

    # show remaining unfilled
    remaining = re.findall(r"\\RESULTVAL\{([^}]+)\}", new_tex)
    if remaining:
        print(f"Still unfilled: {remaining}")
    else:
        print("All placeholders filled!")


if __name__ == "__main__":
    main()
