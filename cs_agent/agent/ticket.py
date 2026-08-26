"""工单存储（内存实现，生产环境可替换为数据库）。"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ..schemas import Ticket, TicketStatus


class TicketStore:
    def __init__(self):
        self._tickets: dict[str, Ticket] = {}
        self._lock = threading.Lock()

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
        return ticket

    def get(self, tid: str) -> Optional[Ticket]:
        return self._tickets.get(tid)

    def list(self) -> List[Ticket]:
        return list(self._tickets.values())

    def close(self, tid: str) -> Optional[Ticket]:
        t = self._tickets.get(tid)
        if t is not None:
            t.status = TicketStatus.CLOSED
        return t
