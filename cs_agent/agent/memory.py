"""用户长期记忆存储（JSON 文件），跨会话按 user_id 记住历史问答，重启不丢。"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    MAX_MEMORIES = 50  # 每个用户最多保留的记忆条数

    def __init__(self, path, max_memories: int = MAX_MEMORIES):
        self._path = Path(path)
        self._max = max_memories
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, user_id: str, question: str, answer: str) -> None:
        """记录一轮问答作为记忆；每个用户只保留最近 N 条。"""
        with self._lock:
            user = self._data.setdefault(user_id, {"memories": []})
            user["memories"].append({"question": question, "answer": answer, "time": _now()})
            user["memories"] = user["memories"][-self._max:]
            user["updated_at"] = _now()
            self._save()

    def get(self, user_id: str, limit: int = 5) -> List[dict]:
        """返回该用户最近 limit 条记忆（新的在前），[{question, answer, time}]。"""
        user = self._data.get(user_id)
        if not user:
            return []
        mems = user.get("memories", [])
        return list(reversed(mems[-limit:]))

    def list_users(self) -> List[dict]:
        out = []
        for uid, user in self._data.items():
            out.append(
                {
                    "user_id": uid,
                    "memory_count": len(user.get("memories", [])),
                    "updated_at": user.get("updated_at"),
                }
            )
        out.sort(key=lambda x: x["updated_at"] or "", reverse=True)
        return out
