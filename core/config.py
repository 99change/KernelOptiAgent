import os
from dataclasses import dataclass

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class LLMConfig:
    """LLM 客户端配置（仅支持 Qwen / Dashscope）"""
    model: str = "qwen3.5-flash-2026-02-23"
    temperature: float = 0.2
    max_tokens: int = 4096
    api_key: str = ""

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


@dataclass
class SystemConfig:
    """系统级配置"""
    gpu_device: int = 0
    max_optimization_rounds: int = 5
    min_improvement_threshold: float = 0.05   # 5% 以上才算有效优化
    output_dir: str = "./results"
    enable_logging: bool = True
    # 是否跳过真实 GPU 测评（用于没有 GPU 的环境，改用 LLM 估算）
    mock_profiling: bool = False


# 全局单例配置，可被 main.py 覆盖
LLM_CONFIG = LLMConfig()
SYS_CONFIG = SystemConfig()
