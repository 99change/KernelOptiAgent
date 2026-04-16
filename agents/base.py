"""
base.py
所有 Agent 的基类，封装 LLM 调用和工具调用。
"""

import json
import logging
from typing import Any

from openai import OpenAI

from core.config import LLMConfig, LLM_CONFIG
from core.memory import AgentMemory


class BaseAgent:
    """
    Agent 基类：
    - 持有 LLM 客户端
    - 提供 _think() 调用 LLM
    - 提供 _store_memory() / _retrieve_memory()
    - 子类实现 execute()
    """

    def __init__(self, name: str, llm_config: LLMConfig = LLM_CONFIG):
        self.name = name
        self.memory = AgentMemory()
        self._llm_config = llm_config
        self._client = self._build_client(llm_config)
        self.logger = logging.getLogger(f"Agent.{name}")

    # ─────────────────────────────────────────
    # LLM 调用
    # ─────────────────────────────────────────

    def _think(self, prompt: str, expect_json: bool = False) -> Any:
        """
        调用 LLM，返回文本或解析好的 dict（expect_json=True 时）。
        """
        self.logger.debug(f"[{self.name}] Sending prompt ({len(prompt)} chars)")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert CUDA GPU kernel optimization assistant. "
                    "You produce concise, correct, and practical answers. "
                    "When asked for JSON, output only valid JSON with no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._llm_config.model,
            messages=messages,
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
        )

        content = response.choices[0].message.content.strip()

        if expect_json:
            return self._parse_json(content)
        return content

    # ─────────────────────────────────────────
    # 记忆管理
    # ─────────────────────────────────────────

    def _store_memory(self, key: str, value: Any) -> None:
        self.memory.save(key, value)

    def _retrieve_memory(self, key: str) -> Any:
        return self.memory.retrieve(key)

    # ─────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────

    def _build_client(self, cfg: LLMConfig) -> OpenAI:
        kwargs = {"api_key": cfg.api_key}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        return OpenAI(**kwargs)

    def _parse_json(self, text: str) -> Any:
        """从 LLM 输出中提取 JSON，兼容带 ```json 代码块的情况"""
        # 去掉 markdown 代码块
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse failed: {e}\nRaw text: {text[:300]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement execute()")
