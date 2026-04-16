"""
analyzer.py
分析 CUDA kernel，找出瓶颈和优化策略。
"""

from core.models import AnalysisResult
from core.config import LLM_CONFIG
from agents.base import BaseAgent
from tools.kernel_tools import analyze_syntax, detect_memory_pattern, estimate_parallelism


class AnalyzerAgent(BaseAgent):

    def __init__(self, llm_config=LLM_CONFIG):
        super().__init__("AnalyzerAgent", llm_config)

    def execute(self, kernel_code: str) -> AnalysisResult:
        self.logger.info("Starting kernel analysis...")

        # 1. 静态分析工具（不需要 LLM，纯代码解析）
        syntax_info = analyze_syntax(kernel_code)
        memory_pattern = detect_memory_pattern(kernel_code)
        parallelism = estimate_parallelism(kernel_code)

        # 2. 构建提示词，让 LLM 基于静态分析结果做深度分析
        prompt = f"""
Analyze this CUDA kernel and identify performance bottlenecks and optimization strategies.

## Static Analysis Results (already computed):
- Kernel count: {syntax_info['kernel_count']}
- Uses shared memory: {syntax_info['has_shared_memory']}
- Uses atomics: {syntax_info['has_atomics']}
- Loop depth: {syntax_info['loop_depth']}
- Memory access pattern: {memory_pattern}
- Parallelism config: {parallelism}

## Kernel Code:
```cuda
{kernel_code}
```

Based on the code and static analysis, identify:
1. The main performance bottlenecks
2. Concrete optimization strategies to apply

Return ONLY a JSON object in this exact format:
{{
    "bottlenecks": ["bottleneck1", "bottleneck2"],
    "optimization_strategies": ["strategy1", "strategy2", "strategy3"]
}}

Strategies should be specific and actionable, for example:
- "use shared memory tiling to reduce global memory access"
- "coalesce memory access by reordering thread access pattern"
- "increase thread block size to improve occupancy"
- "unroll inner loop to reduce loop overhead"
- "use warp-level primitives for reduction"
"""

        # 3. LLM 分析
        analysis = self._think(prompt, expect_json=True)

        bottlenecks = analysis.get("bottlenecks", [])
        strategies = analysis.get("optimization_strategies", [])

        self.logger.info(f"Found {len(bottlenecks)} bottlenecks, {len(strategies)} strategies")

        return AnalysisResult(
            bottlenecks=bottlenecks,
            strategies=strategies,
            code_snippet=kernel_code,
            raw_analysis=str(analysis),
        )
