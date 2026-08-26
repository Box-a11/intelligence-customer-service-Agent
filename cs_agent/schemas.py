"""数据模型（Pydantic）。"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    QA = "qa"
    CONSULTATION = "consultation"
    COMPLAINT = "complaint"
    UNCLEAR = "unclear"


class TicketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class RetrievedDoc(BaseModel):
    source: str
    content: str
    score: float


class Ticket(BaseModel):
    id: str
    intent: str
    user_message: str
    summary: str
    status: TicketStatus = TicketStatus.OPEN
    created_at: str


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    needs_clarification: bool = False
    ticket: Optional[Ticket] = None
    sources: List[str] = Field(default_factory=list)
