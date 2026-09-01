"""可观测性指标与 /metrics 接口测试。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from cs_agent.agent.metrics import MetricsStore
from cs_agent.api.main import app


def test_metrics_store_records_and_snapshots():
    m = MetricsStore()
    m.record(
        intent="qa", latency_ms=1200, sources=["04-退款说明"],
        needs_clarification=False, has_ticket=False, message="退款多久到账",
    )
    m.record(
        intent="complaint", latency_ms=800, sources=[],
        needs_clarification=False, has_ticket=True, message="我要投诉",
    )
    snap = m.snapshot()
    assert snap["requests"] == 2
    assert snap["intents"] == {"qa": 1, "complaint": 1}
    assert snap["latency"]["avg_ms"] == 1000
    assert snap["latency"]["max_ms"] == 1200
    assert snap["retrieval"]["total_sources"] == 1
    assert snap["retrieval"]["top_sources"][0]["source"] == "04-退款说明"
    # 最近请求按时间倒序，最新的一条排最前
    assert len(snap["recent_requests"]) == 2
    assert snap["recent_requests"][0]["message"] == "我要投诉"


def test_metrics_store_persists(tmp_path):
    p = tmp_path / "metrics.json"
    m = MetricsStore(str(p))
    m.record(
        intent="qa", latency_ms=1200, sources=["04-退款说明"],
        needs_clarification=False, has_ticket=False, message="退款多久到账",
    )
    # 模拟重启：用同一路径重新实例化，累计指标应恢复
    m2 = MetricsStore(str(p))
    snap = m2.snapshot()
    assert snap["requests"] == 1
    assert snap["intents"] == {"qa": 1}
    assert snap["latency"]["avg_ms"] == 1200
    assert snap["retrieval"]["top_sources"][0]["source"] == "04-退款说明"


def test_metrics_endpoint_shape():
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    d = r.json()
    assert {"system", "counts", "tickets_breakdown", "intents", "latency", "retrieval", "recent_requests"} <= set(d)
    assert "llm" in d["system"]
    assert "requests" in d["counts"]
    assert "avg_ms" in d["latency"]


def test_dashboard_page():
    client = TestClient(app)
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "监控大盘" in r.text
