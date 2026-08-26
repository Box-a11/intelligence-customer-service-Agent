"""FastAPI 接口测试。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from cs_agent.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_qa():
    r = client.post("/chat", json={"message": "运费怎么算", "session_id": "api-qa"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "qa"
    assert data["reply"]
    assert data["sources"]


def test_chat_complaint():
    r = client.post("/chat", json={"message": "我要投诉你们服务", "session_id": "api-complaint"})
    data = r.json()
    assert data["intent"] == "complaint"
    assert data["ticket"] is not None


def test_chat_clarify():
    r = client.post("/chat", json={"message": "这个", "session_id": "api-unclear"})
    data = r.json()
    assert data["intent"] == "unclear"
    assert data["needs_clarification"] is True


def test_tickets_list():
    r = client.get("/tickets")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_sessions_persistence_and_query():
    # 通过 /chat 产生会话历史
    client.post("/chat", json={"message": "运费怎么算", "session_id": "hist-1"})
    # 会话列表
    r = client.get("/sessions")
    assert r.status_code == 200
    sessions = r.json()
    assert any(s["session_id"] == "hist-1" for s in sessions)
    # 单会话历史
    r2 = client.get("/sessions/hist-1")
    assert r2.status_code == 200
    msgs = r2.json()["messages"]
    assert any(m["role"] == "assistant" for m in msgs)


def test_memory_across_sessions():
    uid = "mem-user-1"
    # 同一用户、不同会话，各问一个问题
    client.post("/chat", json={"message": "退款多久到账", "session_id": "s-1", "user_id": uid})
    client.post("/chat", json={"message": "偏远地区运费怎么算", "session_id": "s-2", "user_id": uid})
    r = client.get("/memory/" + uid)
    assert r.status_code == 200
    mems = r.json()["memories"]
    assert len(mems) >= 2
