"""会话历史持久化存储（JSON 文件），按 session_id 记录对话消息，服务重启不丢。"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, path):
        self._path = Path(path)
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

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息（role 为 user / assistant）。"""
        with self._lock:
            sess = self._data.setdefault(session_id, {"messages": []})
            sess["messages"].append({"role": role, "content": content, "time": _now()})
            sess["updated_at"] = _now()
            self._save()

    def get(self, session_id: str) -> List[dict]:
        """返回某会话的消息列表 [{"role","content"}]，不含时间戳。"""
        sess = self._data.get(session_id)
        if not sess:
            return []
        return [{"role": m["role"], "content": m["content"]} for m in sess["messages"]]

    def list_sessions(self) -> List[dict]:
        """列出所有会话摘要（含首条用户消息预览），按最近更新倒序。"""
        out = []
        for sid, sess in self._data.items():
            msgs = sess.get("messages", [])
            preview = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            out.append(
                {
                    "session_id": sid,
                    "message_count": len(msgs),
                    "preview": preview[:40],
                    "updated_at": sess.get("updated_at"),
                }
            )
        out.sort(key=lambda x: x["updated_at"] or "", reverse=True)
        return out
