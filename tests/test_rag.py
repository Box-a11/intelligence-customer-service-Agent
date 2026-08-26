"""RAG 检索测试。"""
from __future__ import annotations


def test_retriever_builds_and_searches(mock_agent):
    retriever = mock_agent["retriever"]
    assert len(retriever.chunks) >= 10
    docs = retriever.search("偏远地区运费怎么算")
    assert docs
    assert docs[0].score > 0
    assert "02-配送与物流" in [d.source for d in docs]


def test_retriever_relevant_source(mock_agent):
    retriever = mock_agent["retriever"]
    assert "03-退换货政策" in [d.source for d in retriever.search("怎么申请退货")]
    assert "05-发票说明" in [d.source for d in retriever.search("电子发票怎么开")]
    assert "06-会员与积分" in [d.source for d in retriever.search("会员积分怎么获得")]


def test_retriever_unrelated_query_scores_low(mock_agent):
    retriever = mock_agent["retriever"]
    related = retriever.search("运费怎么算")[0].score
    unrelated = retriever.search("为什么天空是蓝色的")[0].score
    assert related > unrelated


def test_router_narrows_search_to_relevant_shard(mock_agent):
    retriever = mock_agent["retriever"]
    # 发票问题只应命中发票分片，不混入其他文档
    docs = retriever.search("电子发票怎么开")
    assert docs
    assert all(d.source == "05-发票说明" for d in docs)
