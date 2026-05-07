from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

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
    "compute_underutilized":   "Increase arithmetic intensity by loop unrolling (#pragma unroll), ILP (each thread handles multiple elements), or fusing adjacent element-wise operations into one kernel. Do NOT use tensor cores unless the kernel already performs matrix multiply.",
    "shared_memory_underused": "Tile global memory accesses through shared memory to exploit data reuse",
    "memory_latency_bound":    "Hide memory latency using __ldg() read-only cache, software pipelining with register double-buffering, or cuda::memcpy_async. Never use __builtin_prefetch (host-only). Increase ILP so warps can hide latency.",
}


# ─────────────────────────────────────────────
# 硬件 Profiling 数据结构
# ─────────────────────────────────────────────

@dataclass
class PtxasInfo:
    """nvcc --ptxas-options=-v 编译期输出，每个 kernel 取最大值"""
    registers: int = 0          # 每线程寄存器数
    smem_bytes: int = 0         # 静态 shared memory 字节
    spill_stores: int = 0       # local memory spill stores（越大越差）
    spill_loads: int = 0        # local memory spill loads
    available: bool = False     # 是否成功解析


@dataclass
class NcuMetrics:
    """ncu 运行时采集的 GPU Speed Of Light 指标"""
    memory_throughput_pct: float = 0.0    # 整体内存吞吐 % of peak
    compute_throughput_pct: float = 0.0   # SM 计算吞吐 % of peak
    dram_throughput_pct: float = 0.0      # DRAM 带宽 % of peak
    l2_throughput_pct: float = 0.0        # L2 Cache 吞吐 % of peak
    l1_throughput_pct: float = 0.0        # L1/TEX Cache 吞吐 % of peak
    achieved_occupancy_pct: float = 0.0   # 实际 warp occupancy %（from Occupancy section）
    elapsed_cycles: int = 0               # kernel elapsed cycles
    available: bool = False               # 是否成功采集


@dataclass
class HardwareProfile:
    """完整的硬件 profiling 数据（编译期 + 运行时）"""
    exec_time_ms: float = 0.0
    ptxas: PtxasInfo = field(default_factory=PtxasInfo)
    ncu: NcuMetrics = field(default_factory=NcuMetrics)

    def summary(self) -> str:
        """返回供 LLM prompt 使用的简洁文本摘要"""
        lines = [f"  Execution time: {self.exec_time_ms:.3f} ms"]
        if self.ptxas.available:
            lines += [
                f"  Registers per thread: {self.ptxas.registers}",
                f"  Static shared memory: {self.ptxas.smem_bytes} bytes",
                f"  Spill stores: {self.ptxas.spill_stores} bytes",
                f"  Spill loads:  {self.ptxas.spill_loads} bytes",
            ]
        if self.ncu.available:
            lines += [
                f"  Memory Throughput:       {self.ncu.memory_throughput_pct:.1f}% of peak",
                f"  Compute (SM) Throughput: {self.ncu.compute_throughput_pct:.1f}% of peak",
                f"  DRAM Throughput:         {self.ncu.dram_throughput_pct:.1f}% of peak",
                f"  L2 Cache Throughput:     {self.ncu.l2_throughput_pct:.1f}% of peak",
                f"  L1/TEX Throughput:       {self.ncu.l1_throughput_pct:.1f}% of peak",
                f"  Achieved Occupancy:      {self.ncu.achieved_occupancy_pct:.1f}%",
            ]
        return "\n".join(lines)


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
    hardware_profile: "HardwareProfile" = None  # 来自 Profiler 的硬件数据


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
    baseline_time_ms: float
    bottleneck_description: str = ""  # 已废弃，保留仅用于兼容；瓶颈分析由 AnalyzerAgent 负责
    hardware_profile: "HardwareProfile" = None  # 完整硬件数据，传给 AnalyzerAgent


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
