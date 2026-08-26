"""混合检索器：稠密向量余弦 + 查询词面覆盖，召回 Top-K；支持分库分索引（路由器）。

- 稠密向量：provider.embed()（真实 embedding 接口，或离线哈希嵌入回退）
- 词面覆盖：查询 bigram 被文档覆盖的比例，用于稳健判别相关/无关

最终得分 = 0.3 * 余弦相似度(归一化到 [0,1]) + 0.7 * 词面覆盖度

词面覆盖度对「相关 / 无关」判别性强（离线哈希嵌入下余弦有固定基线噪声），故占主导；
余弦用于在候选文档间做语义排序。

分库分索引：传入 `router` 后，build() 按 source 把文档分成若干分片、各自建独立子索引；
search() 先用 router.route(query) 定位相关分片，只在这些分片上召回（无命中回退全量）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .chunker import chunk_text
from .embeddings import coverage
from ..llm.base import LLMProvider
from ..schemas import RetrievedDoc


class Retriever:
    def __init__(
        self,
        provider: LLMProvider,
        top_k: int = 3,
        threshold: float = 0.25,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        embedder=None,
        router=None,
    ):
        self.provider = provider
        self.top_k = top_k
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 可选：独立 embedding 器（如本地 bge 模型）；为 None 时回退 provider.embed()。
        self.embedder = embedder
        # 可选：分片路由器；为 None 时退化为单一全量索引。
        self.router = router
        self.chunks: List[dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._shard_chunks: Dict[str, List[dict]] = {}
        self._shard_matrix: Dict[str, np.ndarray] = {}

    def _embed(self, texts: List[str]) -> list:
        if self.embedder is not None:
            return self.embedder.embed(texts)
        return self.provider.embed(texts)

    def _make_matrix(self, chunks: List[dict]) -> np.ndarray:
        vectors = self._embed([c["content"] for c in chunks]) if chunks else []
        return np.asarray(vectors, dtype=float) if vectors else np.zeros((0, 0))

    def build(self, docs: List[dict]) -> int:
        """对知识库文档分块并建立向量索引，返回分块数量。

        传入 router 时按 source 分片建多个子索引；否则建单一全量索引。
        """
        chunks: List[dict] = []
        for doc in docs:
            for text in chunk_text(doc["content"], self.chunk_size, self.chunk_overlap):
                chunks.append({"source": doc["source"], "content": text})
        self.chunks = chunks

        if self.router is not None:
            grouped: Dict[str, List[dict]] = {}
            for c in chunks:
                grouped.setdefault(c["source"], []).append(c)
            self._shard_chunks = {src: cs for src, cs in grouped.items()}
            self._shard_matrix = {src: self._make_matrix(cs) for src, cs in grouped.items()}
            self._matrix = None
        else:
            self._matrix = self._make_matrix(chunks)
        return len(chunks)

    def _search_index(
        self, query: str, chunks: List[dict], matrix: np.ndarray, limit: int
    ) -> List[RetrievedDoc]:
        """在给定 chunks/matrix 上检索，返回按分数降序的 top-`limit` 条。"""
        if not chunks or matrix.size == 0:
            return []
        qv = np.asarray(self._embed([query])[0], dtype=float)
        cosine = matrix @ qv  # 向量已 L2 归一化，点积即余弦相似度
        cosine_norm = (cosine + 1.0) / 2.0  # 归一化到 [0,1]
        lex = np.array([coverage(query, c["content"]) for c in chunks])
        scores = 0.3 * cosine_norm + 0.7 * lex

        order = np.argsort(-scores)[:limit]
        return [
            RetrievedDoc(
                source=chunks[i]["source"],
                content=chunks[i]["content"],
                score=float(scores[i]),
            )
            for i in order
        ]

    def _search_shards(self, query: str, shards: List[str], limit: int) -> List[RetrievedDoc]:
        """在指定分片上检索，合并去重后按分数降序返回。"""
        merged: Dict[tuple, RetrievedDoc] = {}
        for shard in shards:
            chunks = self._shard_chunks.get(shard)
            matrix = self._shard_matrix.get(shard)
            if chunks is None or matrix is None:
                continue
            for d in self._search_index(query, chunks, matrix, self.top_k):
                key = (d.source, d.content)
                if key not in merged or d.score > merged[key].score:
                    merged[key] = d
        docs = sorted(merged.values(), key=lambda d: d.score, reverse=True)
        return docs[:limit]

    def search(self, query: str) -> List[RetrievedDoc]:
        if self.router is not None:
            # 路由器定位相关分片；无命中回退全量（保持转人工判定等既有行为）
            shards = self.router.route(query) or list(self._shard_chunks)
            return self._search_shards(query, shards, self.top_k)
        if not self.chunks or self._matrix is None or self._matrix.size == 0:
            return []
        return self._search_index(query, self.chunks, self._matrix, self.top_k)

    def search_many(self, queries: List[str], limit: Optional[int] = None) -> List[RetrievedDoc]:
        """多查询召回（多跳问题）：对每个子查询分别检索（内部各自路由），合并去重后按分数降序。

        多跳问题（答案需结合多个知识库片段）会被拆成若干子查询分别召回再合并，
        保证不同文档的相关片段都进入上下文；去重键为 (source, content)，同片段保留最高分。
        单查询时等价于 search()。
        """
        merged: Dict[tuple, RetrievedDoc] = {}
        for q in queries:
            for d in self.search(q):
                key = (d.source, d.content)
                if key not in merged or d.score > merged[key].score:
                    merged[key] = d
        docs = sorted(merged.values(), key=lambda d: d.score, reverse=True)
        cap = limit if limit is not None else self.top_k * max(1, len(queries))
        return docs[:cap]
