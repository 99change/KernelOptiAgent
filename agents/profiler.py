"""
profiler.py
对 CUDA kernel 进行基准性能测评，获取基准时间。
支持真实 GPU 运行和 mock 模式（无 GPU 时使用）。
"""

from core.models import ProfileResult, KernelMetrics
from core.config import LLM_CONFIG, SYS_CONFIG
from agents.base import BaseAgent
from tools.kernel_tools import compile_cuda, run_compiled_kernel, mock_profile


class ProfilerAgent(BaseAgent):

    def __init__(self, llm_config=LLM_CONFIG, mock_mode: bool = None):
        super().__init__("ProfilerAgent", llm_config)
        # mock_mode 优先级：参数 > SYS_CONFIG > False
        if mock_mode is None:
            mock_mode = SYS_CONFIG.mock_profiling
        self.mock_mode = mock_mode

    def execute(self, kernel_code: str) -> ProfileResult:
        self.logger.info(f"Profiling kernel (mock_mode={self.mock_mode})...")

        if self.mock_mode:
            test_result = mock_profile(kernel_code)
        else:
            # 真实编译 + 运行
            compile_result = compile_cuda(kernel_code)
            if not compile_result.success:
                self.logger.warning(
                    f"Compilation failed, falling back to mock profiling.\n"
                    f"Error: {compile_result.error}"
                )
                test_result = mock_profile(kernel_code)
            else:
                test_result = run_compiled_kernel(compile_result.binary_path)

        avg_time = test_result.exec_time_ms if test_result.success else 0.0

        # 让 LLM 解读性能数据，给出瓶颈描述
        prompt = f"""
You are a CUDA performance expert.

A kernel has been profiled with the following results:
- Execution time: {avg_time:.2f} ms
- Profiling mode: {"mock/simulated" if self.mock_mode else "real GPU"}

Based on the kernel code below, describe in ONE short sentence what the most likely performance bottleneck is:

```cuda
{kernel_code[:1500]}
```

Reply with just the bottleneck description, no extra formatting.
"""
        bottleneck_desc = self._think(prompt, expect_json=False)

        self.logger.info(f"Baseline time: {avg_time:.2f} ms")

        return ProfileResult(
            metrics=test_result.metrics or KernelMetrics(exec_time_ms=avg_time),
            bottleneck_description=bottleneck_desc.strip(),
            baseline_time_ms=avg_time,
        )
