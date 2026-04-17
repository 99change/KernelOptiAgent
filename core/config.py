import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 客户端配置"""
    # 支持 openai / github_copilot / qwen
    provider: str = "openai"
    model: str = ""  # 如果为空，根据 provider 选择默认模型
    temperature: float = 0.2
    max_tokens: int = 4096
    # API key 从环境变量读取，不在代码中硬写
    api_key: str = ""
    base_url: str = ""

    def __post_init__(self):
        # 如果未指定模型，根据 provider 选择默认值
        if not self.model:
            if self.provider == "openai":
                self.model = "gpt-4o"
            elif self.provider == "github_copilot":
                self.model = "gpt-4"
            elif self.provider == "qwen":
                self.model = "qwen-max"
        
        # 读取 API key
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY", "")
            elif self.provider == "github_copilot":
                self.api_key = os.environ.get("GITHUB_TOKEN", "")
            elif self.provider == "qwen":
                self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        
        # 设置 base_url
        if not self.base_url:
            if self.provider == "github_copilot":
                self.base_url = "https://api.githubcopilot.com"
            elif self.provider == "qwen":
                self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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
