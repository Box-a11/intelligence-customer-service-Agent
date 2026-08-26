"""用户长期记忆存储测试。"""
from __future__ import annotations

from cs_agent.agent.memory import MemoryStore


def test_add_and_get(tmp_path):
    s = MemoryStore(tmp_path / "mem.json")
    s.add("u1", "退款多久到账", "1-3个工作日")
    s.add("u1", "那银行卡呢", "3-7个工作日")
    mems = s.get("u1")
    assert len(mems) == 2
    # 新的在前
    assert mems[0]["question"] == "那银行卡呢"
    assert mems[1]["question"] == "退款多久到账"


def test_persists_across_reload(tmp_path):
    path = tmp_path / "mem.json"
    s1 = MemoryStore(path)
    s1.add("u1", "怎么退货", "七天无理由")
    s2 = MemoryStore(path)
    assert s2.get("u1")[0]["answer"] == "七天无理由"


def test_get_missing_returns_empty(tmp_path):
    s = MemoryStore(tmp_path / "mem.json")
    assert s.get("nobody") == []


def test_capped_memories(tmp_path):
    s = MemoryStore(tmp_path / "mem.json", max_memories=3)
    for i in range(5):
        s.add("u1", f"q{i}", f"a{i}")
    mems = s.get("u1", limit=10)
    assert len(mems) == 3
    assert mems[0]["question"] == "q4"
