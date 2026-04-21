"""
optimizer.py
逐个尝试优化策略，编译+测评，保留最优版本。
接受结构化 BottleneckIR，构建 hardware-aware 的优化 prompt。
"""

from typing import List, Dict, Optional

from core.models import OptimizationResult, OptimizationHistory, BottleneckItem
from core.config import LLM_CONFIG, SYS_CONFIG
from agents.base import BaseAgent
from tools.kernel_tools import compile_and_test, mock_profile
from tools.knowledge_retrieval import retrieve as retrieve_knowledge


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
        bottleneck_ir: Optional[Dict[str, BottleneckItem]] = None,
    ) -> OptimizationResult:

        self.logger.info(f"Starting optimization with {len(strategies)} strategies...")

        best_code = kernel_code
        best_time = baseline_time_ms
        history: List[OptimizationHistory] = []

        for i, strategy in enumerate(strategies):
            self.logger.info(f"  [{i+1}/{len(strategies)}] Trying: {strategy}")

            # 1. 让 LLM 生成优化后的代码（传入结构化 IR 以构建 hardware-aware prompt）
            optimized_code = self._generate_optimized_code(best_code, strategy, bottleneck_ir)
            if not optimized_code:
                self.logger.warning(f"    LLM returned empty code for strategy: {strategy}")
                continue

            # 2. 编译 + 测评（支持 self-repair 最多 2 次）
            MAX_REPAIR = 2
            test_result = None
            for repair_attempt in range(MAX_REPAIR + 1):
                try:
                    if self.mock_mode:
                        test_result = mock_profile(optimized_code)
                    else:
                        test_result = compile_and_test(optimized_code)
                except Exception as e:
                    self.logger.warning(f"    Test failed: {e}")
                    break

                if test_result.success:
                    break

                # 编译失败 → self-repair
                if repair_attempt < MAX_REPAIR:
                    self.logger.info(
                        f"    Compile error (attempt {repair_attempt+1}/{MAX_REPAIR}), trying self-repair..."
                    )
                    repaired = self._repair_code(optimized_code, test_result.error)
                    if repaired:
                        optimized_code = repaired
                    else:
                        self.logger.warning("    Self-repair returned empty code, giving up")
                        break
                else:
                    self.logger.warning(
                        f"    Compile/run failed after {MAX_REPAIR} repair attempts: {test_result.error[:120]}"
                    )

            if test_result is None or not test_result.success:
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

            # 异常检测：如果结果比 baseline 慢 10 倍以上，直接拒绝
            if baseline_time_ms > 0 and exec_time > baseline_time_ms * 10:
                self.logger.warning(
                    f"    ✗ Anomalous slowdown detected: {exec_time:.2f}ms vs baseline {baseline_time_ms:.2f}ms "
                    f"({exec_time/baseline_time_ms:.0f}x slower). Skipping."
                )
                history.append(OptimizationHistory(
                    strategy=strategy,
                    speedup=0.0,
                    exec_time_ms=exec_time,
                    code=optimized_code,
                    success=False,
                ))
                continue

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

        # ── E2E pass：无约束让 LLM 自由优化，与策略结果竞争 ──────
        self.logger.info("  [E2E] Trying: end-to-end unconstrained optimization")
        e2e_code = self._generate_e2e_code(kernel_code)
        if e2e_code:
            try:
                e2e_result = mock_profile(e2e_code) if self.mock_mode else compile_and_test(e2e_code)
            except Exception as e:
                e2e_result = None
                self.logger.warning(f"    E2E test failed: {e}")

            if e2e_result and e2e_result.success:
                e2e_time = e2e_result.exec_time_ms
                if baseline_time_ms > 0 and e2e_time <= baseline_time_ms * 10:
                    e2e_improvement = (best_time - e2e_time) / best_time if best_time > 0 else 0.0
                    e2e_vs_baseline = (baseline_time_ms - e2e_time) / baseline_time_ms * 100
                    self.logger.info(f"    Time: {e2e_time:.2f} ms  vs baseline: {e2e_vs_baseline:+.1f}%")
                    history.append(OptimizationHistory(
                        strategy="[E2E] unconstrained",
                        speedup=e2e_improvement,
                        exec_time_ms=e2e_time,
                        code=e2e_code,
                        success=True,
                    ))
                    if e2e_time < best_time:
                        best_code = e2e_code
                        best_time = e2e_time
                        self.logger.info("    ✓ E2E result is best so far")
                    else:
                        self.logger.info("    ~ E2E result not better than strategy pass")
                else:
                    self.logger.warning("    ✗ E2E anomalous slowdown, skipped")
            else:
                err = e2e_result.error[:80] if e2e_result else "unknown"
                self.logger.warning(f"    E2E compile/run failed: {err}")
        else:
            self.logger.warning("    E2E LLM returned empty code")

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

    def _generate_optimized_code(
        self,
        kernel_code: str,
        strategy: str,
        bottleneck_ir: Optional[Dict[str, BottleneckItem]] = None,
    ) -> str:
        # 1. 知识库检索相关示例
        example_code = retrieve_knowledge(strategy)
        knowledge_section = ""
        if example_code:
            knowledge_section = f"""
## Reference Example (correct CUDA implementation pattern):
```cuda
{example_code}
```
Study the above example carefully, especially the correct API usage and syntax.

"""

        # 2. 从 BottleneckIR 构建 hardware-aware context（按 score 降序列出）
        ir_section = ""
        if bottleneck_ir:
            active = sorted(
                [(k, v) for k, v in bottleneck_ir.items() if v.score >= 0.3],
                key=lambda x: x[1].score,
                reverse=True,
            )
            if active:
                lines = ["## Detected Bottlenecks (from hardware analysis):"]
                for key, item in active:
                    ev_str = ", ".join(f"{k}={v}" for k, v in item.evidence.items()) if item.evidence else ""
                    ev_part = f" [{ev_str}]" if ev_str else ""
                    lines.append(f"- {key.replace('_', ' ')} (score={item.score:.2f}){ev_part}")

                # 推断约束
                constraints = []
                sm_item = bottleneck_ir.get("shared_memory_underused")
                if sm_item and sm_item.score < 0.5:
                    constraints.append("shared memory already in use — stay within 48 KB")
                reg_item = bottleneck_ir.get("high_register_pressure")
                if reg_item and reg_item.score >= 0.5:
                    constraints.append("register pressure is high — avoid adding new variables")
                if constraints:
                    lines.append("\n## Constraints:")
                    for c in constraints:
                        lines.append(f"- {c}")

                ir_section = "\n".join(lines) + "\n"

        prompt = f"""
You are a CUDA expert. Apply the following optimization to the CUDA kernel below.

## Optimization Goal:
{strategy}

{ir_section}{knowledge_section}## Current Kernel (may already contain prior optimizations — preserve them):
```cuda
{kernel_code}
```

## Requirements:
- Apply ONLY the specified optimization strategy ON TOP of any existing optimizations already present in the kernel above
- Do NOT remove or undo any optimizations already in the code (e.g. shared memory tiling, loop unrolling)
- Keep the kernel semantically correct (same output for same input)
- Return ONLY the complete optimized .cu source code, no explanation
- The code must be compilable with nvcc
- Include necessary headers (#include <cuda_runtime.h> etc.)
- Follow the exact syntax patterns shown in the reference example above

Return just the raw code, no markdown fences.
"""
        result = self._think(prompt, expect_json=False)

        # 清理 LLM 可能包裹的代码块
        if result.startswith("```"):
            lines = result.splitlines()
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            result = "\n".join(inner)

        return result.strip()

    def _repair_code(self, broken_code: str, error_msg: str) -> str:
        """Self-repair: feed nvcc error back to LLM and ask it to fix the code."""
        prompt = f"""You are a CUDA expert. The following CUDA code failed to compile with nvcc.
Fix ONLY the compilation errors. Do not change the optimization logic or algorithm.

## Compiler Error:
```
{error_msg[:1500]}
```

## Broken Code:
```cuda
{broken_code}
```

## Requirements:
- Fix all compilation errors shown above
- Preserve all existing optimizations (shared memory tiling, loop unrolling, etc.)
- Keep the kernel semantically correct (same output for same input)
- Return ONLY the complete fixed .cu source code, no explanation, no markdown fences
"""
        result = self._think(prompt, expect_json=False)
        if result.startswith("```"):
            lines = result.splitlines()
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            result = "\n".join(inner)
        return result.strip()

    def _generate_e2e_code(self, kernel_code: str) -> str:
        """无约束端到端优化：不注入策略/知识库，让 LLM 自由发挥。"""
        prompt = f"""You are a CUDA optimization expert. Rewrite the following CUDA kernel to be as fast as possible on a modern NVIDIA GPU (sm_120).

Rules:
- Output ONLY the complete, compilable CUDA C++ source file. No markdown, no explanation.
- Keep the same function signature and behavior (same output for same input).
- Use any optimization you judge appropriate.

Original kernel:
{kernel_code}
"""
        result = self._think(prompt, expect_json=False)
        if result.startswith("```"):
            lines = result.splitlines()
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            result = "\n".join(inner)
        return result.strip()
