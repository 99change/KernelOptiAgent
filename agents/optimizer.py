"""
optimizer.py
逐个尝试优化策略，编译+测评，保留最优版本。
"""

from typing import List

from core.models import OptimizationResult, OptimizationHistory
from core.config import LLM_CONFIG, SYS_CONFIG
from agents.base import BaseAgent
from tools.kernel_tools import compile_and_test, mock_profile


class OptimizerAgent(BaseAgent):

    def __init__(self, llm_config=LLM_CONFIG, mock_mode: bool = None):
        super().__init__("OptimizerAgent", llm_config)
        if mock_mode is None:
            mock_mode = SYS_CONFIG.mock_profiling
        self.mock_mode = mock_mode
        self.min_improvement = SYS_CONFIG.min_improvement_threshold

    def execute(
        self,
        kernel_code: str,
        strategies: List[str],
        baseline_time_ms: float,
    ) -> OptimizationResult:

        self.logger.info(f"Starting optimization with {len(strategies)} strategies...")

        best_code = kernel_code
        best_time = baseline_time_ms
        history: List[OptimizationHistory] = []

        for i, strategy in enumerate(strategies):
            self.logger.info(f"  [{i+1}/{len(strategies)}] Trying: {strategy}")

            # 1. 让 LLM 生成优化后的代码
            optimized_code = self._generate_optimized_code(best_code, strategy)
            if not optimized_code:
                self.logger.warning(f"    LLM returned empty code for strategy: {strategy}")
                continue

            # 2. 编译 + 测评（或 mock）
            try:
                if self.mock_mode:
                    test_result = mock_profile(optimized_code)
                else:
                    test_result = compile_and_test(optimized_code)
            except Exception as e:
                self.logger.warning(f"    Test failed: {e}")
                history.append(OptimizationHistory(
                    strategy=strategy,
                    speedup=0.0,
                    exec_time_ms=0.0,
                    code=optimized_code,
                    success=False,
                ))
                continue

            if not test_result.success:
                self.logger.warning(f"    Compile/run failed: {test_result.error}")
                history.append(OptimizationHistory(
                    strategy=strategy,
                    speedup=0.0,
                    exec_time_ms=0.0,
                    code=optimized_code,
                    success=False,
                ))
                continue

            # 3. 计算提升
            exec_time = test_result.exec_time_ms
            if best_time > 0:
                improvement = (best_time - exec_time) / best_time
            else:
                improvement = 0.0

            record = OptimizationHistory(
                strategy=strategy,
                speedup=improvement,
                exec_time_ms=exec_time,
                code=optimized_code,
                success=True,
            )
            history.append(record)

            self.logger.info(f"    Time: {exec_time:.2f} ms  Improvement: {improvement*100:.1f}%")

            # 4. 决策：超过阈值才保留
            if improvement > self.min_improvement:
                best_code = optimized_code
                best_time = exec_time
                self._store_memory(f"strategy_{i}_{strategy[:20]}", optimized_code)
                self.logger.info(f"    ✓ Accepted (improvement={improvement*100:.1f}%)")
            else:
                self.logger.info(f"    ✗ Rejected (below threshold {self.min_improvement*100:.0f}%)")

        # 计算相对于原始 baseline 的总加速比
        if baseline_time_ms > 0:
            total_speedup = (baseline_time_ms - best_time) / baseline_time_ms
        else:
            total_speedup = 0.0

        self.logger.info(f"Optimization complete. Total speedup: {total_speedup*100:.1f}%")

        return OptimizationResult(
            optimized_code=best_code,
            speedup=total_speedup,
            history=history,
        )

    # ─────────────────────────────────────────
    # 内部：让 LLM 生成优化代码
    # ─────────────────────────────────────────

    def _generate_optimized_code(self, kernel_code: str, strategy: str) -> str:
        prompt = f"""
You are a CUDA expert. Apply the following optimization to the CUDA kernel below.

## Optimization Strategy:
{strategy}

## Original Kernel:
```cuda
{kernel_code}
```

## Requirements:
- Apply ONLY the specified optimization strategy
- Keep the kernel semantically correct (same output for same input)
- Return ONLY the complete optimized .cu source code, no explanation
- The code must be compilable with nvcc
- Include necessary headers (#include <cuda_runtime.h> etc.)

Return just the raw code, no markdown fences.
"""
        result = self._think(prompt, expect_json=False)

        # 清理 LLM 可能包裹的代码块
        if result.startswith("```"):
            lines = result.splitlines()
            # 去掉第一行 (```cuda 或 ```) 和最后一行 (```)
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            result = "\n".join(inner)

        return result.strip()
