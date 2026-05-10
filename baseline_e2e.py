"""
baseline_e2e.py
端到端基准测试：完全抛弃分析/知识库/策略，直接让 LLM 优化 kernel，看裸效果。

用法：
    python baseline_e2e.py --input examples/vector_add.cu --model qwen3.5-flash-2026-02-23
"""

import argparse
import os
import sys

from openai import OpenAI
from core.config import DASHSCOPE_BASE_URL, LLMConfig
from tools.kernel_tools import compile_and_test

# ─────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a CUDA optimization expert. Below is a CUDA kernel implementation.
Your task: rewrite it to be as fast as possible on a modern NVIDIA GPU (sm_120).

Rules:
- Output ONLY the complete, compilable CUDA C++ source file. No markdown, no explanation.
- Keep the same function signature and behavior (same inputs/outputs, same correctness).
- Use any optimization you judge appropriate.

Original kernel:
{code}
"""
# ─────────────────────────────────────────


def call_llm(client: OpenAI, model: str, code: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=4096,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(code=code)}],
    )
    text = resp.choices[0].message.content or ""
    # 去掉 LLM 可能包裹的 markdown 代码块
    if "```" in text:
        lines = text.splitlines()
        inside = False
        result = []
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                result.append(line)
        text = "\n".join(result)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="qwen-max")
    parser.add_argument("--tries", type=int, default=3, help="重试次数")
    args = parser.parse_args()

    with open(args.input) as f:
        original_code = f.read()

    cfg = LLMConfig(model=args.model)
    if not cfg.api_key:
        print("ERROR: DASHSCOPE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=cfg.api_key, base_url=DASHSCOPE_BASE_URL, timeout=120.0)

    # 先测 baseline
    print(f"[Baseline] Compiling and timing original kernel...")
    baseline = compile_and_test(original_code)
    if not baseline.success:
        print(f"ERROR: Baseline compile failed: {baseline.error}")
        sys.exit(1)
    print(f"[Baseline] {baseline.exec_time_ms:.3f} ms\n")

    best_time = baseline.exec_time_ms
    best_code = original_code
    best_try = 0

    for i in range(1, args.tries + 1):
        print(f"[Try {i}/{args.tries}] Calling LLM...")
        optimized = call_llm(client, args.model, original_code)

        if not optimized:
            print(f"  LLM returned empty output, skipping.")
            continue

        result = compile_and_test(optimized)
        if not result.success:
            print(f"  Compile/run failed: {result.error[:120]}")
            continue

        speedup = (baseline.exec_time_ms - result.exec_time_ms) / baseline.exec_time_ms * 100
        print(f"  Time: {result.exec_time_ms:.3f} ms  ({speedup:+.1f}% vs baseline)")

        if result.exec_time_ms < best_time:
            best_time = result.exec_time_ms
            best_code = optimized
            best_try = i

    print(f"\n{'='*50}")
    overall = (baseline.exec_time_ms - best_time) / baseline.exec_time_ms * 100
    print(f"  Baseline  : {baseline.exec_time_ms:.3f} ms")
    print(f"  Best      : {best_time:.3f} ms  (try {best_try})")
    print(f"  Speedup   : {overall:+.1f}%")
    print(f"{'='*50}")

    out_path = "results/e2e_optimized.cu"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(best_code)
    print(f"  Output    : {out_path}")


if __name__ == "__main__":
    main()
