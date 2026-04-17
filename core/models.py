from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# 固定 bottleneck 类型 schema（不可随意增删）
BOTTLENECK_SCHEMA = [
    "non_coalesced_memory",
    "memory_bound",
    "low_occupancy",
    "high_register_pressure",
    "warp_divergence",
    "compute_underutilized",
    "shared_memory_underused",
    "memory_latency_bound",
]

# bottleneck → 默认优化方向（供 OptimizerAgent 构建 prompt 用）
BOTTLENECK_STRATEGIES = {
    "non_coalesced_memory":    "Coalesce memory access so consecutive threads access consecutive addresses",
    "memory_bound":            "Use float4 vectorized loads and __ldg() to increase memory throughput",
    "low_occupancy":           "Tune block size and reduce register/shared memory usage to raise occupancy",
    "high_register_pressure":  "Reduce register usage via variable reuse or __launch_bounds__ directive",
    "warp_divergence":         "Eliminate branch divergence within warps by restructuring conditionals",
    "compute_underutilized":   "Increase arithmetic intensity by fusing operations or using tensor cores",
    "shared_memory_underused": "Tile global memory accesses through shared memory to exploit data reuse",
    "memory_latency_bound":    "Hide memory latency using __ldg() read-only cache, software pipelining with register double-buffering, or cuda::memcpy_async. Never use __builtin_prefetch (host-only). Increase ILP so warps can hide latency.",
}


@dataclass
class BottleneckItem:
    """单个瓶颈的结构化表示"""
    score: float                              # 严重程度 0.0（无）→ 1.0（极严重）
    evidence: Dict[str, Any] = field(default_factory=dict)  # 来自代码/profiling 的证据


@dataclass
class AnalysisResult:
    """代码分析结果"""
    bottlenecks: List[str]                    # 人类可读描述（兼容旧代码）
    strategies: List[str]                     # 从 IR 推导出的优化方向
    code_snippet: str
    raw_analysis: str = ""
    bottleneck_ir: Dict[str, BottleneckItem] = field(default_factory=dict)  # 结构化 IR


@dataclass
class KernelMetrics:
    """Kernel 性能指标"""
    exec_time_ms: float = 0.0
    memory_bw_pct: float = 0.0
    register_usage: int = 0
    occupancy: float = 0.0


@dataclass
class ProfileResult:
    """性能测评结果"""
    metrics: KernelMetrics
    bottleneck_description: str
    baseline_time_ms: float


@dataclass
class OptimizationHistory:
    """单次优化记录"""
    strategy: str
    speedup: float
    exec_time_ms: float
    code: str
    success: bool


@dataclass
class OptimizationResult:
    """优化执行结果"""
    optimized_code: str
    speedup: float
    history: List[OptimizationHistory] = field(default_factory=list)


@dataclass
class OptimizationReport:
    """最终输出报告"""
    original_kernel: str
    optimized_kernel: str
    speedup: float
    strategies_applied: List[str]
    analysis: AnalysisResult
    baseline_time_ms: float
    optimized_time_ms: float
