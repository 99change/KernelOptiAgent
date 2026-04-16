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

from core.models import KernelMetrics


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class CompileResult:
    success: bool
    binary_path: str = ""
    ptx_path: str = ""
    error: str = ""


@dataclass
class TestResult:
    success: bool
    exec_time_ms: float = 0.0
    metrics: Optional[KernelMetrics] = None
    error: str = ""


# ─────────────────────────────────────────────
# 静态分析工具
# ─────────────────────────────────────────────

def analyze_syntax(code: str) -> dict:
    """
    对 CUDA kernel 做轻量级静态分析。
    返回: {"kernel_count": int, "has_shared_memory": bool,
           "has_atomics": bool, "loop_depth": int, "line_count": int}
    """
    result = {
        "kernel_count": len(re.findall(r'__global__\s+\w+\s+\w+\s*\(', code)),
        "has_shared_memory": "__shared__" in code,
        "has_atomics": bool(re.search(r'atomic\w+', code)),
        "has_texture": "tex" in code or "texture" in code,
        "loop_depth": _estimate_loop_depth(code),
        "line_count": len(code.splitlines()),
        "thread_idx_usage": "threadIdx" in code,
        "block_idx_usage": "blockIdx" in code,
    }
    return result


def detect_memory_pattern(code: str) -> str:
    """
    检测内存访问模式，返回描述字符串。
    简单规则：
    - 若访问下标是 threadIdx.x，认为 coalesced
    - 若访问下标含复杂乘法，认为 strided
    - 其他认为 unknown
    """
    # 检测最常见的 coalesced 模式：arr[threadIdx.x] 或 arr[... + threadIdx.x]
    if re.search(r'\[\s*(?:.*\+\s*)?threadIdx\.x\s*\]', code):
        return "coalesced"
    # 检测 strided：arr[threadIdx.x * N]
    if re.search(r'\[\s*threadIdx\.\w+\s*\*\s*\d+', code):
        return "strided"
    # 检测随机访问
    if re.search(r'\[.*blockIdx.*threadIdx.*\*', code):
        return "possibly_strided"
    return "unknown"


def estimate_parallelism(code: str) -> dict:
    """
    从代码中估计建议的并行度配置。
    如果代码没有明确指定，返回默认建议。
    """
    # 尝试从代码注释或 <<<>>> 中提取
    launch_match = re.search(r'<<<\s*(\w+)\s*,\s*(\w+)\s*>>>', code)
    if launch_match:
        return {
            "grid_dim": launch_match.group(1),
            "block_dim": launch_match.group(2),
            "source": "extracted_from_launch"
        }
    return {
        "grid_dim": "N/A",
        "block_dim": "N/A",
        "source": "not_found_in_code"
    }


# ─────────────────────────────────────────────
# 编译工具
# ─────────────────────────────────────────────

def compile_cuda(code: str, gpu_arch: str = "sm_80") -> CompileResult:
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
            ptx_path=final_ptx
        )


def compile_and_test(code: str, gpu_arch: str = "sm_80") -> TestResult:
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
