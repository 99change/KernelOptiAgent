"""
kernel_tools.py
工具函数库：静态分析 + 编译 + 性能测评

注意：
- 编译和运行工具需要真实的 nvcc 和 CUDA 环境
- 如果设置 mock_profiling=True（core/config.py），则跳过真实 GPU 运行
"""

import re
import os
import subprocess
import tempfile
import time
import json
from dataclasses import dataclass
from typing import Optional

from core.models import KernelMetrics, PtxasInfo, NcuMetrics, HardwareProfile


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class CompileResult:
    success: bool
    binary_path: str = ""
    ptx_path: str = ""
    error: str = ""
    ptxas_info: PtxasInfo = None  # type: ignore

    def __post_init__(self):
        if self.ptxas_info is None:
            self.ptxas_info = PtxasInfo()


@dataclass
class TestResult:
    success: bool
    exec_time_ms: float = 0.0
    metrics: Optional[KernelMetrics] = None
    error: str = ""


# ─────────────────────────────────────────────
# 编译工具
# ─────────────────────────────────────────────

def compile_cuda(code: str, gpu_arch: str = "sm_120") -> CompileResult:
    """
    用 nvcc 编译 CUDA kernel 代码。
    code 应包含完整的可编译 .cu 文件内容。
    """
    # 检查 nvcc 是否可用
    if not _nvcc_available():
        return CompileResult(
            success=False,
            error="nvcc not found. Please install CUDA Toolkit."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.cu")
        out_path = os.path.join(tmpdir, "kernel.out")
        ptx_path = os.path.join(tmpdir, "kernel.ptx")

        with open(src_path, "w") as f:
            f.write(code)

        # 编译成可执行文件
        compile_cmd = [
            "nvcc", src_path,
            "-o", out_path,
            f"-arch={gpu_arch}",
            "-O3",
            "--ptxas-options=-v",
        ]
        ret = subprocess.run(
            compile_cmd,
            capture_output=True, text=True, timeout=60
        )

        if ret.returncode != 0:
            return CompileResult(success=False, error=ret.stderr)

        # 解析 ptxas 编译统计信息（来自 stderr）
        ptxas_info = parse_ptxas_info(ret.stderr)

        # 同时生成 PTX（用于分析寄存器使用等）
        ptx_cmd = [
            "nvcc", src_path,
            "-ptx", "-o", ptx_path,
            f"-arch={gpu_arch}",
        ]
        subprocess.run(ptx_cmd, capture_output=True, timeout=30)

        # 把编译产物复制到持久位置
        persistent_dir = tempfile.mkdtemp(prefix="kernelopt_")
        import shutil
        final_bin = os.path.join(persistent_dir, "kernel.out")
        shutil.copy(out_path, final_bin)
        final_ptx = ""
        if os.path.exists(ptx_path):
            final_ptx = os.path.join(persistent_dir, "kernel.ptx")
            shutil.copy(ptx_path, final_ptx)

        return CompileResult(
            success=True,
            binary_path=final_bin,
            ptx_path=final_ptx,
            ptxas_info=ptxas_info,
        )


def compile_and_test(code: str, gpu_arch: str = "sm_120") -> TestResult:
    """编译 + 运行 + 返回时间"""
    compile_result = compile_cuda(code, gpu_arch)
    if not compile_result.success:
        return TestResult(success=False, error=compile_result.error)
    return run_compiled_kernel(compile_result.binary_path)


# ─────────────────────────────────────────────
# 测评工具
# ─────────────────────────────────────────────

def run_compiled_kernel(binary_path: str, num_runs: int = 3) -> TestResult:
    """
    运行已编译的 kernel，返回平均执行时间。
    binary 需要能独立运行（main 函数自带计时逻辑）。
    """
    if not os.path.exists(binary_path):
        return TestResult(success=False, error=f"Binary not found: {binary_path}")

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ret = subprocess.run(
            [binary_path],
            capture_output=True, text=True, timeout=30
        )
        elapsed = (time.perf_counter() - start) * 1000  # ms

        if ret.returncode != 0:
            return TestResult(success=False, error=ret.stderr)

        # 尝试从程序输出解析时间（如果 kernel 打印了 "time: X ms"）
        parsed_time = _parse_time_from_output(ret.stdout)
        times.append(parsed_time if parsed_time else elapsed)

    avg_time = sum(times) / len(times)
    return TestResult(
        success=True,
        exec_time_ms=avg_time,
        metrics=KernelMetrics(exec_time_ms=avg_time)
    )


def mock_profile(code: str) -> TestResult:
    """
    在没有 GPU 的环境下，用代码特征估算一个模拟时间。
    仅用于开发测试，不反映真实性能。
    """
    lines = len(code.splitlines())
    loops = _estimate_loop_depth(code)
    # 纯粹的模拟值，没有任何工程意义
    simulated_time = 10.0 + lines * 0.05 + loops * 2.0
    return TestResult(
        success=True,
        exec_time_ms=round(simulated_time, 2),
        metrics=KernelMetrics(exec_time_ms=simulated_time)
    )


def validate_correctness(original_code: str, optimized_code: str) -> bool:
    """
    验证优化后代码的正确性。
    当前策略：只检查基本语法合法性（能否编译）。
    更严格的语义验证需要用户提供测试数据。
    """
    result = compile_cuda(optimized_code)
    return result.success


# ─────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────

def _nvcc_available() -> bool:
    try:
        subprocess.run(["nvcc", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _estimate_loop_depth(code: str) -> int:
    """粗略统计 for/while 嵌套深度"""
    max_depth = 0
    depth = 0
    for line in code.splitlines():
        stripped = line.strip()
        if re.match(r'^(for|while)\s*\(', stripped):
            depth += 1
            max_depth = max(max_depth, depth)
        if stripped == "}":
            depth = max(0, depth - 1)
    return max_depth


def _parse_time_from_output(output: str) -> Optional[float]:
    """从程序输出中解析 'time: X ms' 或 'elapsed: X ms' 格式的时间"""
    match = re.search(r'(?:time|elapsed)[:\s]+([0-9.]+)\s*ms', output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# ─────────────────────────────────────────────
# ptxas 信息解析
# ─────────────────────────────────────────────

def parse_ptxas_info(stderr: str) -> PtxasInfo:
    """
    解析 nvcc --ptxas-options=-v 的 stderr 输出，提取所有 kernel 的最大资源用量。

    典型输出格式：
      ptxas info    : Used 32 registers, 360 bytes smem, 336 bytes cmem[0]
      ptxas info    : Function properties for vectorAdd
                      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    """
    max_regs = 0
    total_smem = 0
    total_spill_stores = 0
    total_spill_loads = 0
    found = False

    for line in stderr.splitlines():
        # 解析 "Used N registers, M bytes smem"
        m = re.search(r'Used\s+(\d+)\s+registers', line)
        if m:
            max_regs = max(max_regs, int(m.group(1)))
            found = True

        m = re.search(r'(\d+)\s+bytes\s+smem', line)
        if m:
            total_smem = max(total_smem, int(m.group(1)))

        # 解析 "N bytes stack frame, M bytes spill stores, K bytes spill loads"
        m = re.search(r'(\d+)\s+bytes\s+spill\s+stores', line)
        if m:
            total_spill_stores += int(m.group(1))

        m = re.search(r'(\d+)\s+bytes\s+spill\s+loads', line)
        if m:
            total_spill_loads += int(m.group(1))

    return PtxasInfo(
        registers=max_regs,
        smem_bytes=total_smem,
        spill_stores=total_spill_stores,
        spill_loads=total_spill_loads,
        available=found,
    )


# ─────────────────────────────────────────────
# ncu 运行时 profiling
# ─────────────────────────────────────────────

_NCU_BINARY = "ncu"

# 要采集的 ncu 指标（来自 GPU Speed Of Light Throughput + Occupancy sections）
_NCU_METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
])


def run_ncu_profile(binary_path: str, timeout: int = 120) -> NcuMetrics:
    """
    用 ncu 对已编译的 CUDA binary 做一次 profiling，返回关键 GPU 指标。
    只 profile 第一个 kernel 调用（--kernel-id ::1）以节省时间。
    失败时返回 available=False 的 NcuMetrics。
    """
    if not os.path.exists(binary_path):
        return NcuMetrics()

    try:
        cmd = [
            _NCU_BINARY,
            "--target-processes", "all",
            "--metrics", _NCU_METRICS,
            "--print-summary", "per-kernel",
            binary_path,
        ]
        ret = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout
        )
        output = ret.stdout + ret.stderr
        return _parse_ncu_metrics_output(output)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return NcuMetrics()


def _parse_ncu_metrics_output(text: str) -> NcuMetrics:
    """
    解析 ncu --metrics 的文本输出，提取各项百分比指标。
    ncu 输出格式（每行一个指标）：
      sm__throughput.avg.pct_of_peak_sustained_elapsed    %    56.81
    也可能是 summary 格式，需要灵活处理。
    """
    values: dict = {}

    # ncu --metrics 输出格式：metric_name  unit  value（以空格对齐）
    metric_patterns = {
        "compute": r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)",
        "memory": r"gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)",
        "dram": r"dram__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)",
        "l2": r"lts__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)",
        "l1": r"l1tex__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)",
        "occupancy": r"sm__warps_active\.avg\.pct_of_peak_sustained_active\s+%\s+([\d.]+)",
    }

    for key, pattern in metric_patterns.items():
        m = re.search(pattern, text)
        if m:
            values[key] = float(m.group(1))

    if not values:
        return NcuMetrics()

    return NcuMetrics(
        compute_throughput_pct=values.get("compute", 0.0),
        memory_throughput_pct=values.get("memory", 0.0),
        dram_throughput_pct=values.get("dram", 0.0),
        l2_throughput_pct=values.get("l2", 0.0),
        l1_throughput_pct=values.get("l1", 0.0),
        achieved_occupancy_pct=values.get("occupancy", 0.0),
        available=True,
    )


def compile_and_full_profile(code: str, gpu_arch: str = "sm_120") -> HardwareProfile:
    """
    完整 profiling pipeline：编译 → 计时 → ncu。
    返回 HardwareProfile（包含 ptxas 编译期信息 + ncu 运行时信息 + 执行时间）。
    失败时返回 available=False 的子字段，exec_time_ms=0。
    """
    hw = HardwareProfile()

    # 1. 编译（含 ptxas 解析）
    compile_result = compile_cuda(code, gpu_arch)
    if not compile_result.success:
        return hw

    hw.ptxas = compile_result.ptxas_info

    # 2. 计时
    test_result = run_compiled_kernel(compile_result.binary_path)
    if test_result.success:
        hw.exec_time_ms = test_result.exec_time_ms

    # 3. ncu profiling
    hw.ncu = run_ncu_profile(compile_result.binary_path)

    return hw
