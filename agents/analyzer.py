"""
analyzer.py
分析 CUDA kernel，输出结构化 bottleneck IR（固定 schema + score + evidence）。
LLM 只负责"填表"，不做自由文本分析。多次运行后对 score 取平均（aggregation）。
硬件数字（ptxas + ncu）由 ProfilerAgent 采集，是 LLM 打分的唯一依据。
"""

from typing import Dict, List, Any

from core.models import (
    AnalysisResult, BottleneckItem, BOTTLENECK_SCHEMA, BOTTLENECK_STRATEGIES,
    HardwareProfile,
)
from core.config import LLM_CONFIG
from agents.base import BaseAgent

# 同一输入运行 LLM 的次数，取 score 均值以提升稳定性
_N_AGGREGATIONS = 3
# 列入 strategies 的 score 阈值
_SCORE_THRESHOLD = 0.4


class AnalyzerAgent(BaseAgent):

    def __init__(self, llm_config=LLM_CONFIG):
        super().__init__("AnalyzerAgent", llm_config)

    def execute(self, kernel_code: str, hardware_profile: HardwareProfile) -> AnalysisResult:
        self.logger.info("Starting kernel analysis (hardware-grounded IR mode)...")

        # 硬件 profiling 摘要
        hw_ctx = hardware_profile.summary()

        # 自动注释：bound 类型 + occupancy 警告
        mem = hardware_profile.ncu.memory_throughput_pct
        cmp = hardware_profile.ncu.compute_throughput_pct
        occ = hardware_profile.ncu.achieved_occupancy_pct
        regs = hardware_profile.ptxas.registers
        hw_notes = f"  bound_type: {'memory-bound' if mem > cmp else 'compute-bound'}\n"
        hw_notes += f"  registers/thread: {regs} {'(HIGH, likely limits occupancy)' if regs > 64 else '(normal)'}\n"
        if occ < 25:
            hw_notes += f"  occupancy: {occ:.1f}% — CRITICALLY LOW\n"
        elif occ < 50:
            hw_notes += f"  occupancy: {occ:.1f}% — LOW\n"

        # 构建"填表"prompt
        schema_example = "\n".join(
            f'  "{k}": {{"score": <0.0-1.0>, "evidence": {{...}}}},'
            for k in BOTTLENECK_SCHEMA
        )

        prompt = f"""You are a CUDA performance expert acting as a structured form filler.

## Measured Hardware Profile (ground truth — trust these numbers, not code guesses):
{hw_ctx}
{hw_notes}
## Kernel Code:
```cuda
{kernel_code}
```

Fill in the bottleneck assessment form. Assign score (0.0 = not present, 1.0 = severe).
Base scores on the measured hardware data above. Put actual measured numbers in evidence fields.

Return ONLY a valid JSON object with EXACTLY these keys (no extra keys, no explanation text):
{{
{schema_example}
}}

Evidence field format (use measured values):
- non_coalesced_memory: {{"access_pattern": "coalesced/strided/random"}}
- memory_bound: {{"memory_throughput_pct": {mem:.1f}, "dram_throughput_pct": {hardware_profile.ncu.dram_throughput_pct:.1f}}}
- low_occupancy: {{"achieved_occupancy_pct": {occ:.1f}, "registers": {regs}}}
- high_register_pressure: {{"registers": {regs}, "spill_stores": {hardware_profile.ptxas.spill_stores}}}
- warp_divergence: {{"branches_in_kernel": true/false}}
- compute_underutilized: {{"compute_throughput_pct": {cmp:.1f}}}
- shared_memory_underused: {{"smem_bytes": {hardware_profile.ptxas.smem_bytes}, "data_reuse_possible": true/false}}
- memory_latency_bound: {{"memory_throughput_pct": {mem:.1f}, "compute_throughput_pct": {cmp:.1f}}}
"""

        # 聚合：跑 N 次，对每个 bottleneck 的 score 取均值
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

        # 构建 BottleneckIR（平均 score）
        bottleneck_ir: Dict[str, BottleneckItem] = {}
        for key in BOTTLENECK_SCHEMA:
            scores = raw_scores[key]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            bottleneck_ir[key] = BottleneckItem(
                score=round(avg_score, 3),
                evidence=last_evidence[key],
            )

        # 从 IR 推导人类可读描述和优化策略（按 score 排序）
        sorted_items = sorted(
            bottleneck_ir.items(), key=lambda x: x[1].score, reverse=True
        )

        bottlenecks: List[str] = []
        strategies: List[str] = []
        for key, item in sorted_items:
            if item.score >= _SCORE_THRESHOLD:
                ev = item.evidence
                ev_str = ", ".join(f"{k}={v}" for k, v in ev.items()) if ev else ""
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
            hardware_profile=hardware_profile,
        )

