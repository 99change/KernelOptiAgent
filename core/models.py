from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AnalysisResult:
    """代码分析结果"""
    bottlenecks: List[str]
    strategies: List[str]
    code_snippet: str
    raw_analysis: str = ""


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
