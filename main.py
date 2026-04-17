"""
main.py
GPU Kernel 优化 Agent 主入口

用法：
    python main.py --input examples/vector_add.cu
    python main.py --input examples/vector_add.cu --provider qwen --model qwen-max
    python main.py --input examples/vector_add.cu --mock     # 无 GPU 时使用 mock 模式
    python main.py --input examples/vector_add.cu --rounds 3 # 限制优化轮数
    
环境变量：
    export OPENAI_API_KEY=sk-xxx                             # OpenAI
    export DASHSCOPE_API_KEY=sk-xxx                          # Qwen (阿里云)
    export GITHUB_TOKEN=ghp_xxx                              # GitHub Copilot
"""

import os
import sys
import argparse
import logging

from core.config import LLM_CONFIG, SYS_CONFIG, LLMConfig, SystemConfig
from core.models import OptimizationReport
from agents.analyzer import AnalyzerAgent
from agents.profiler import ProfilerAgent
from agents.optimizer import OptimizerAgent


# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ─────────────────────────────────────────────
# 核心流程
# ─────────────────────────────────────────────

def run(kernel_code: str, mock: bool = False, max_rounds: int = 5, llm_config: LLMConfig = None) -> OptimizationReport:
    """
    主优化流程：分析 → 测评 → 优化 → 返回报告

    Args:
        kernel_code: 输入的 CUDA kernel 源代码
        mock: True 则跳过真实 GPU 编译，使用模拟测评
        max_rounds: 最大优化策略尝试次数
        llm_config: LLM 配置对象（如不提供则使用全局 LLM_CONFIG）

    Returns:
        OptimizationReport 包含优化后代码和所有过程数据
    """
    if llm_config is None:
        llm_config = LLM_CONFIG
    
    SYS_CONFIG.mock_profiling = mock
    SYS_CONFIG.max_optimization_rounds = max_rounds

    separator = "=" * 60

    # ── Phase 1: 分析 ──────────────────────────────────────────
    print(f"\n{separator}")
    print("  Phase 1/3 : Analyzing Kernel")
    print(separator)

    analyzer = AnalyzerAgent(llm_config=llm_config)
    analysis = analyzer.execute(kernel_code)

    print(f"  Bottlenecks found    : {len(analysis.bottlenecks)}")
    for b in analysis.bottlenecks:
        print(f"    - {b}")
    print(f"  Strategies proposed  : {len(analysis.strategies)}")
    for s in analysis.strategies:
        print(f"    - {s}")

    # ── Phase 2: 基准测评 ─────────────────────────────────────
    print(f"\n{separator}")
    print("  Phase 2/3 : Profiling Baseline")
    print(separator)

    profiler = ProfilerAgent(llm_config=llm_config, mock_mode=mock)
    profile = profiler.execute(kernel_code)

    print(f"  Baseline time        : {profile.baseline_time_ms:.2f} ms")
    print(f"  Bottleneck           : {profile.bottleneck_description}")

    # ── Phase 3: 优化 ─────────────────────────────────────────
    print(f"\n{separator}")
    print("  Phase 3/3 : Optimizing")
    print(separator)

    strategies = analysis.strategies[:max_rounds]
    optimizer = OptimizerAgent(llm_config=llm_config, mock_mode=mock)
    optimization = optimizer.execute(
        kernel_code=kernel_code,
        strategies=strategies,
        baseline_time_ms=profile.baseline_time_ms,
    )

    successful = [h for h in optimization.history if h.success]
    print(f"\n  Strategies tried     : {len(optimization.history)}")
    print(f"  Strategies succeeded : {len(successful)}")
    print(f"  Total speedup        : {optimization.speedup * 100:.1f}%")

    # ── 生成报告 ──────────────────────────────────────────────
    optimized_time = profile.baseline_time_ms * (1 - optimization.speedup)
    report = OptimizationReport(
        original_kernel=kernel_code,
        optimized_kernel=optimization.optimized_code,
        speedup=optimization.speedup,
        strategies_applied=[h.strategy for h in successful],
        analysis=analysis,
        baseline_time_ms=profile.baseline_time_ms,
        optimized_time_ms=optimized_time,
    )

    return report


def save_report(report: OptimizationReport, output_dir: str):
    """将报告和优化结果写入文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 优化后的 kernel
    kernel_path = os.path.join(output_dir, "optimized_kernel.cu")
    with open(kernel_path, "w") as f:
        f.write(report.optimized_kernel)

    # 2. 文本报告
    report_path = os.path.join(output_dir, "optimization_report.txt")
    with open(report_path, "w") as f:
        f.write("GPU Kernel Optimization Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Baseline time    : {report.baseline_time_ms:.2f} ms\n")
        f.write(f"Optimized time   : {report.optimized_time_ms:.2f} ms\n")
        f.write(f"Speedup          : {report.speedup * 100:.1f}%\n\n")
        f.write("Bottlenecks identified:\n")
        for b in report.analysis.bottlenecks:
            f.write(f"  - {b}\n")
        f.write("\nStrategies applied:\n")
        for s in report.strategies_applied:
            f.write(f"  - {s}\n")

    return kernel_path, report_path


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU Kernel Optimization Agent"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CUDA kernel file (.cu)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./results",
        help="Output directory (default: ./results)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock profiling (no real GPU required)",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=5,
        help="Max optimization rounds (default: 5)",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "github_copilot", "qwen"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: from config)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # 读取输入 kernel
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r") as f:
        kernel_code = f.read()

    if not kernel_code.strip():
        print("Error: Input file is empty.", file=sys.stderr)
        sys.exit(1)

    # 根据 provider 和 model 创建新的 LLM 配置
    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model or LLM_CONFIG.model,  # 用参数或默认值
    )

    # 检查 API key
    if not llm_config.api_key:
        print(
            "Error: No API key found.\n"
            "  OpenAI:         export OPENAI_API_KEY=sk-xxx\n"
            "  GitHub Copilot: export GITHUB_TOKEN=ghp_xxx  (provider=github_copilot)\n"
            "  Qwen:           export DASHSCOPE_API_KEY=sk-xxx  (provider=qwen)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nInput    : {args.input}")
    print(f"Output   : {args.output}")
    print(f"Provider : {llm_config.provider}")
    print(f"Model    : {llm_config.model}")
    print(f"Mock     : {args.mock}")
    print(f"Rounds   : {args.rounds}")

    # 运行优化
    report = run(
        kernel_code=kernel_code,
        mock=args.mock,
        max_rounds=args.rounds,
        llm_config=llm_config,
    )

    # 保存结果
    kernel_path, report_path = save_report(report, args.output)

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")
    print(f"  Speedup        : {report.speedup * 100:.1f}%")
    print(f"  Optimized code : {kernel_path}")
    print(f"  Report         : {report_path}")
    print()


if __name__ == "__main__":
    main()
