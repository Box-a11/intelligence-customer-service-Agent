"""分库分索引路由器测试。"""
from __future__ import annotations

from cs_agent.rag.router import Router


def test_route_single_shard():
    r = Router()
    assert r.route("电子发票怎么开") == ["05-发票说明"]


def test_route_multihop_hits_multiple_shards():
    r = Router()
    shards = r.route("退货后退款多久到账")
    assert "03-退换货政策" in shards
    assert "04-退款说明" in shards


def test_route_no_match_returns_empty():
    r = Router()
    assert r.route("为什么天空是蓝色的") == []


def test_route_ranks_by_keyword_hits():
    r = Router()
    # 「退款」「到账」同属退款说明，命中数最高，应排最前
    shards = r.route("退款到账")
    assert shards and shards[0] == "04-退款说明"


def test_route_respects_top_n():
    r = Router(top_n=1)
    # 命中多个分片时只保留 top_n
    assert len(r.route("退货退款")) == 1
