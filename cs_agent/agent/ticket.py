"""工单存储（JSON 文件持久化），重启不丢。"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..schemas import Ticket, TicketStatus


class TicketStore:
    def __init__(self, path: str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._tickets: dict[str, Ticket] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for tid, item in data.items():
                    self._tickets[tid] = Ticket(**item)
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {tid: t.model_dump() for tid, t in self._tickets.items()}
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def create(self, intent: str, user_message: str, summary: str) -> Ticket:
        tid = "TK-" + uuid.uuid4().hex[:8].upper()
        ticket = Ticket(
            id=tid,
            intent=intent,
            user_message=user_message,
            summary=summary,
            status=TicketStatus.OPEN,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._tickets[tid] = ticket
            self._save()
        return ticket

    def get(self, tid: str) -> Optional[Ticket]:
        return self._tickets.get(tid)

    def list(self) -> List[Ticket]:
        return list(self._tickets.values())

    def close(self, tid: str) -> Optional[Ticket]:
        t = self._tickets.get(tid)
        if t is not None:
            t.status = TicketStatus.CLOSED
            with self._lock:
                self._save()
        return t
