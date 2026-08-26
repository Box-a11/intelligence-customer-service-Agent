"""LangGraph 端到端路由测试。"""
from __future__ import annotations


def test_qa_routes_to_answer(mock_agent):
    agent = mock_agent["agent"]
    r = agent.run("请问运费怎么算", "s-qa")
    assert r.get("intent") == "qa"
    assert r.get("ticket") is None
    assert r.get("answer")
    assert r.get("needs_clarification") is False
    assert r.get("sources")


def test_consultation_routes_to_answer(mock_agent):
    agent = mock_agent["agent"]
    r = agent.run("你们支持分期付款吗", "s-consult")
    assert r.get("intent") == "consultation"
    assert r.get("ticket") is None
    assert r.get("answer")


def test_complaint_routes_to_ticket(mock_agent):
    agent = mock_agent["agent"]
    r = agent.run("你们服务太差了，我要投诉", "s-complaint")
    assert r.get("intent") == "complaint"
    assert r.get("ticket") is not None
    assert "工单号" in r.get("answer", "")


def test_unclear_routes_to_clarify(mock_agent):
    agent = mock_agent["agent"]
    r = agent.run("这个", "s-unclear")
    assert r.get("intent") == "unclear"
    assert r.get("needs_clarification") is True
    assert r.get("ticket") is None


def test_unanswerable_escalates_to_ticket(mock_agent):
    agent = mock_agent["agent"]
    r = agent.run("为什么天空是蓝色的", "s-esc")
    assert r.get("ticket") is not None


def test_react_multi_round_loop(mock_agent):
    agent = mock_agent["agent"]
    r = agent.run("偏远地区运费怎么算")
    assert r.get("intent") == "qa"
    # 经历了「检索 → 再思考 → 回答」的多轮循环
    assert r.get("react_round", 0) >= 1
    assert r.get("answer")
    assert r.get("sources")


def test_multi_hop_retrieves_multiple_sources(mock_agent):
    agent = mock_agent["agent"]
    r = agent.run("退货后退款多久到账", "s-multihop")
    assert r.get("intent") == "qa"
    assert r.get("ticket") is None
    # 多跳问题需结合「退换货政策」与「退款说明」两份文档
    assert "03-退换货政策" in r["sources"]
    assert "04-退款说明" in r["sources"]
    assert r.get("answer")


def test_multi_turn_uses_history(mock_agent):
    agent = mock_agent["agent"]
    r1 = agent.run("你好")
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": r1["answer"]},
    ]
    # 传入历史后仍基于最新消息正确路由
    r2 = agent.run("我要投诉你们服务", history=history)
    assert r2.get("intent") == "complaint"
    assert r2.get("ticket") is not None


def test_no_stale_ticket_across_turns(mock_agent):
    agent = mock_agent["agent"]
    # 第一轮：投诉生成工单
    r1 = agent.run("你们服务太差了，我要投诉")
    assert r1.get("ticket") is not None
    history = [
        {"role": "user", "content": "你们服务太差了，我要投诉"},
        {"role": "assistant", "content": r1["answer"]},
    ]
    # 第二轮：普通问答不应携带上一轮的工单
    r2 = agent.run("偏远地区运费怎么算", history=history)
    assert r2.get("ticket") is None
    assert r2.get("intent") == "qa"
    assert r2.get("answer")
