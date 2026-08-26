"""会话历史持久化测试。"""
from __future__ import annotations

from cs_agent.agent.history import ConversationStore


def test_append_and_get(tmp_path):
    s = ConversationStore(tmp_path / "conv.json")
    s.append("u1", "user", "你好")
    s.append("u1", "assistant", "您好")
    msgs = s.get("u1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "你好"


def test_persists_across_reload(tmp_path):
    path = tmp_path / "conv.json"
    s1 = ConversationStore(path)
    s1.append("u1", "user", "退款多久到账")
    s2 = ConversationStore(path)  # 重新加载，模拟服务重启
    assert s2.get("u1")[0]["content"] == "退款多久到账"


def test_list_sessions(tmp_path):
    s = ConversationStore(tmp_path / "conv.json")
    s.append("a", "user", "x")
    s.append("b", "user", "y")
    sessions = s.list_sessions()
    assert {x["session_id"] for x in sessions} == {"a", "b"}
    assert all(x["message_count"] == 1 for x in sessions)
    assert all(x["preview"] for x in sessions)  # 首条用户消息作为预览


def test_get_missing_returns_empty(tmp_path):
    s = ConversationStore(tmp_path / "conv.json")
    assert s.get("nope") == []
