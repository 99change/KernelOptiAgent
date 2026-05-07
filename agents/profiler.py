"""
profiler.py
对 CUDA kernel 进行基准性能测评，同时采集硬件 profiling 数据。
输出 ProfileResult（含 HardwareProfile），供 AnalyzerAgent 做数据驱动的瓶颈分析。
纯工具流，不调用 LLM。
"""

import logging

from core.models import ProfileResult, KernelMetrics, HardwareProfile
from core.config import SYS_CONFIG
from tools.kernel_tools import (
    compile_cuda, run_compiled_kernel, mock_profile,
    run_ncu_profile, compile_and_full_profile,
)


class ProfilerAgent:

    def __init__(self, mock_mode: bool = None):
        self.logger = logging.getLogger("Agent.ProfilerAgent")
        if mock_mode is None:
            mock_mode = SYS_CONFIG.mock_profiling
        self.mock_mode = mock_mode

    def execute(self, kernel_code: str) -> ProfileResult:
        """
        真实模式：compile → ptxas parse → timing → ncu profile，全部打包进 HardwareProfile。
        mock 模式：跳过所有 GPU 调用，返回模拟时间，hardware_profile=None。
        """
        self.logger.info(f"Profiling kernel (mock_mode={self.mock_mode})...")

        if self.mock_mode:
            test_result = mock_profile(kernel_code)
            avg_time = test_result.exec_time_ms if test_result.success else 0.0
            self.logger.info(f"[mock] Baseline time: {avg_time:.2f} ms")
            return ProfileResult(
                metrics=test_result.metrics or KernelMetrics(exec_time_ms=avg_time),
                baseline_time_ms=avg_time,
                hardware_profile=None,
            )

        # ── 真实模式：完整 hardware profile ───────────────────
        self.logger.info("  [1/3] Compiling + parsing ptxas info...")
        compile_result = compile_cuda(kernel_code)
        if not compile_result.success:
            self.logger.warning(
                f"Compilation failed, falling back to mock profiling.\n"
                f"Error: {compile_result.error}"
            )
            test_result = mock_profile(kernel_code)
            avg_time = test_result.exec_time_ms if test_result.success else 0.0
            return ProfileResult(
                metrics=test_result.metrics or KernelMetrics(exec_time_ms=avg_time),
                baseline_time_ms=avg_time,
                hardware_profile=None,
            )

        ptxas = compile_result.ptxas_info
        if ptxas.available:
            self.logger.info(
                f"  ptxas: registers={ptxas.registers}, smem={ptxas.smem_bytes}B, "
                f"spill_stores={ptxas.spill_stores}B"
            )
        else:
            self.logger.warning("  ptxas: no info extracted from stderr")

        self.logger.info("  [2/3] Timing kernel (3 runs avg)...")
        test_result = run_compiled_kernel(compile_result.binary_path)
        avg_time = test_result.exec_time_ms if test_result.success else 0.0
        self.logger.info(f"  Baseline time: {avg_time:.3f} ms")

        self.logger.info("  [3/3] Running ncu to collect GPU metrics (this may take ~30s)...")
        ncu = run_ncu_profile(compile_result.binary_path)
        if ncu.available:
            self.logger.info(
                f"  ncu: mem_throughput={ncu.memory_throughput_pct:.1f}%, "
                f"compute={ncu.compute_throughput_pct:.1f}%, "
                f"occupancy={ncu.achieved_occupancy_pct:.1f}%"
            )
        else:
            self.logger.warning("  ncu: profiling failed or no metrics extracted")

        hw_profile = HardwareProfile(
            exec_time_ms=avg_time,
            ptxas=ptxas,
            ncu=ncu,
        )

        return ProfileResult(
            metrics=KernelMetrics(exec_time_ms=avg_time),
            baseline_time_ms=avg_time,
            hardware_profile=hw_profile,
        )
