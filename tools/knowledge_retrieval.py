"""
knowledge_retrieval.py
基于关键词匹配，根据优化策略名称检索相关 CUDA 示例代码。

工作原理：
- 维护一张 strategy_keywords → 示例文件 的映射表
- 对输入的 strategy 字符串做小写关键词匹配
- 返回匹配到的示例代码字符串（用于注入 LLM prompt）
"""

import os
from typing import Optional

# 知识库目录（相对本文件的上级目录）
_KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")

# 关键词 → 文件名 映射
# key: 文件名（不含路径）
# value: 触发该文件的关键词列表（小写）
_KEYWORD_MAP: dict[str, list[str]] = {
    "float4_vectorized.cu": [
        "float4", "vectorized", "vector load", "vectorize",
        "128-bit", "16-byte", "memory throughput", "wide load",
        "__ldg", "ldg", "read-only cache", "texture cache",
    ],
    "restrict_qualifiers.cu": [
        "__restrict__", "restrict", "alias", "pointer alias",
        "compiler optimization", "compiler hint",
    ],
    "shared_memory_tiling.cu": [
        "shared memory", "tiling", "tile", "__shared__",
        "data reuse", "cache", "matmul", "matrix",
    ],
    "loop_unrolling.cu": [
        "unroll", "loop unrolling", "#pragma unroll",
        "loop overhead", "instruction level parallelism", "ilp",
    ],
    "occupancy_tuning.cu": [
        "occupancy", "block size", "thread block", "warp",
        "__launch_bounds__", "register pressure", "sm utilization",
        "active warps", "increase occupancy",
    ],
}


def retrieve(strategy: str) -> Optional[str]:
    """
    根据策略描述，返回最相关的 CUDA 示例代码。
    若无匹配，返回 None。

    Args:
        strategy: 优化策略描述字符串，例如
                  "use vectorized memory access (float4) to increase memory throughput"

    Returns:
        示例代码字符串，或 None
    """
    strategy_lower = strategy.lower()

    best_file = None
    best_score = 0

    for filename, keywords in _KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in strategy_lower)
        if score > best_score:
            best_score = score
            best_file = filename

    if best_file is None or best_score == 0:
        return None

    filepath = os.path.join(_KNOWLEDGE_DIR, best_file)
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        return f.read()


def retrieve_all_matching(strategy: str) -> list[str]:
    """
    返回所有匹配（得分 > 0）的示例代码列表。
    用于需要多个示例的场景。
    """
    strategy_lower = strategy.lower()
    results = []

    for filename, keywords in _KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in strategy_lower)
        if score > 0:
            filepath = os.path.join(_KNOWLEDGE_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    results.append(f.read())

    return results


def list_knowledge_files() -> list[str]:
    """列出所有知识库文件名"""
    return list(_KEYWORD_MAP.keys())
