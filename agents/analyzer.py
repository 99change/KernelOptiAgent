"""
analyzer.py
分析 CUDA kernel，输出结构化 bottleneck IR（固定 schema + score + evidence）。
LLM 只负责"填表"，不做自由文本分析。多次运行后对 score 取平均（aggregation）。
"""

from typing import Dict, List, Any

from core.models import AnalysisResult, BottleneckItem, BOTTLENECK_SCHEMA, BOTTLENECK_STRATEGIES
from core.config import LLM_CONFIG
from agents.base import BaseAgent
from tools.kernel_tools import analyze_syntax, detect_memory_pattern, estimate_parallelism

# 同一输入运行 LLM 的次数，取 score 均值以提升稳定性
_N_AGGREGATIONS = 3
# 列入 strategies 的 score 阈值
_SCORE_THRESHOLD = 0.4


class AnalyzerAgent(BaseAgent):

    def __init__(self, llm_config=LLM_CONFIG):
        super().__init__("AnalyzerAgent", llm_config)

    def execute(self, kernel_code: str) -> AnalysisResult:
        self.logger.info("Starting kernel analysis (structured IR mode)...")

        # 1. 静态分析工具（不需要 LLM）
        syntax_info    = analyze_syntax(kernel_code)
        memory_pattern = detect_memory_pattern(kernel_code)
        parallelism    = estimate_parallelism(kernel_code)

        static_ctx = (
            f"- Kernel count: {syntax_info['kernel_count']}\n"
            f"- Uses shared memory: {syntax_info['has_shared_memory']}\n"
            f"- Uses atomics: {syntax_info['has_atomics']}\n"
            f"- Loop depth: {syntax_info['loop_depth']}\n"
            f"- Memory access pattern: {memory_pattern}\n"
            f"- Parallelism config: {parallelism}"
        )

        # 2. 构建"填表"prompt
        schema_example = "\n".join(
            f'  "{k}": {{"score": <0.0-1.0>, "evidence": {{...}}}},'
            for k in BOTTLENECK_SCHEMA
        )
        prompt = f"""You are a CUDA performance expert acting as a structured form filler.

## Static Analysis:
{static_ctx}

## Kernel Code:
```cuda
{kernel_code}
```

Fill in the bottleneck assessment form below.
For each entry assign a score (0.0 = not present, 1.0 = severe) and provide concise evidence
derived only from the code and static analysis above.

Return ONLY a valid JSON object with EXACTLY these keys (no extra keys, no explanation text):
{{
{schema_example}
}}

Evidence field examples:
- non_coalesced_memory: {{"stride": <int>, "access_pattern": "strided/random"}}
- memory_bound: {{"arithmetic_intensity": "low/medium/high", "loads_per_flop": <float>}}
- low_occupancy: {{"block_size": <int>, "registers_estimated": "high/low"}}
- high_register_pressure: {{"loop_depth": <int>, "temp_vars": "many/few"}}
- warp_divergence: {{"branches_in_kernel": true/false, "condition_type": "..."}}
- compute_underutilized: {{"flops_per_element": <float>}}
- shared_memory_underused: {{"data_reuse_possible": true/false}}
- memory_latency_bound: {{"independent_loads": true/false}}
"""

        # 3. 聚合：跑 N 次，对每个 bottleneck 的 score 取均值
        raw_scores: Dict[str, List[float]] = {k: [] for k in BOTTLENECK_SCHEMA}
        last_evidence: Dict[str, Any] = {k: {} for k in BOTTLENECK_SCHEMA}

        for i in range(_N_AGGREGATIONS):
            try:
                result = self._think(prompt, expect_json=True)
                for key in BOTTLENECK_SCHEMA:
                    entry = result.get(key, {})
                    if isinstance(entry, dict):
                        score = float(entry.get("score", 0.0))
                        score = max(0.0, min(1.0, score))
                        raw_scores[key].append(score)
                        if entry.get("evidence"):
                            last_evidence[key] = entry["evidence"]
            except Exception as e:
                self.logger.warning(f"Aggregation run {i+1} failed: {e}")

        # 4. 构建 BottleneckIR（平均 score）
        bottleneck_ir: Dict[str, BottleneckItem] = {}
        for key in BOTTLENECK_SCHEMA:
            scores = raw_scores[key]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            bottleneck_ir[key] = BottleneckItem(
                score=round(avg_score, 3),
                evidence=last_evidence[key],
            )

        # 5. 从 IR 推导人类可读描述和优化策略（按 score 排序）
        sorted_items = sorted(
            bottleneck_ir.items(), key=lambda x: x[1].score, reverse=True
        )

        bottlenecks: List[str] = []
        strategies: List[str] = []
        for key, item in sorted_items:
            if item.score >= _SCORE_THRESHOLD:
                ev = item.evidence
                ev_str = ", ".join(f"{k}={v}" for k, v in ev.items()) if ev else "code analysis"
                bottlenecks.append(f"{key} (score={item.score:.2f}, evidence: {ev_str})")
                strategies.append(BOTTLENECK_STRATEGIES[key])

        self.logger.info(
            f"Found {len(bottlenecks)} active bottlenecks "
            f"(threshold={_SCORE_THRESHOLD}, aggregations={_N_AGGREGATIONS})"
        )

        return AnalysisResult(
            bottlenecks=bottlenecks,
            strategies=strategies,
            code_snippet=kernel_code,
            raw_analysis=str(bottleneck_ir),
            bottleneck_ir=bottleneck_ir,
        )

