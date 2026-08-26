"""工单存储测试。"""
from __future__ import annotations


def test_ticket_lifecycle(mock_agent):
    store = mock_agent["store"]
    t = store.create("complaint", "服务很差", "服务很差")
    assert t.id.startswith("TK-")
    assert t.intent == "complaint"
    assert store.get(t.id) is not None
    assert any(x.id == t.id for x in store.list())
    store.close(t.id)
    assert store.get(t.id).status.value == "closed"


def test_ticket_get_missing_returns_none(mock_agent):
    store = mock_agent["store"]
    assert store.get("TK-NOTEXIST") is None
