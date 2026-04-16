from typing import Any, List, Optional


class AgentMemory:
    """简单的基于 dict 的 Agent 记忆系统"""

    def __init__(self):
        self._storage: dict = {}

    def save(self, key: str, value: Any) -> None:
        self._storage[key] = value

    def retrieve(self, key: str) -> Optional[Any]:
        return self._storage.get(key)

    def search(self, query: str) -> List[Any]:
        """返回所有 key 包含 query 的值"""
        return [v for k, v in self._storage.items() if query.lower() in k.lower()]

    def all_keys(self) -> List[str]:
        return list(self._storage.keys())

    def clear(self) -> None:
        self._storage.clear()
