"""意图分类测试（MockProvider，确定性）。"""
from __future__ import annotations

from cs_agent.llm.mock import MockProvider


def _classify(text: str) -> str:
    return MockProvider().classify_intent([{"role": "user", "content": text}])


def test_qa_intent():
    assert _classify("请问运费怎么算") == "qa"
    assert _classify("退款多久到账") == "qa"
    assert _classify("发票怎么开") == "qa"


def test_consultation_intent():
    assert _classify("你们支持分期付款吗") == "consultation"
    assert _classify("会员怎么开通") == "consultation"
    assert _classify("618有什么优惠活动") == "consultation"


def test_complaint_intent():
    assert _classify("你们服务太差了，我要投诉") == "complaint"
    assert _classify("我要举报商家虚假宣传") == "complaint"


def test_unclear_intent():
    assert _classify("这个") == "unclear"
    assert _classify("有问题") == "unclear"


def test_embed_deterministic_and_normalized():
    p = MockProvider()
    v1 = p.embed(["你好世界"])[0]
    v2 = p.embed(["你好世界"])[0]
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6
