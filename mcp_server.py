"""RAG 知识库 MCP Server：把 cs_agent.rag 的检索能力暴露为 MCP 工具。

供外部 LLM / Agent 直接调用（无需走 HTTP / FastAPI）：
- search / search_many：知识库检索（混合打分 + 可选分片路由）
- route / list_shards：分库分索引路由查询
- health：服务状态

运行方式（stdio 传输，供 MCP 客户端注册）：
    .venv/Scripts/python mcp_server.py

检索器懒加载：首次 search 时才构建索引（本地 bge 首次加载约数秒，之后复用）。
"""
from __future__ import annotations

from typing import List, Optional

from mcp.server.mcpserver import MCPServer

from cs_agent import config
from cs_agent.llm.mock import MockProvider
from cs_agent.rag.loader import load_documents
from cs_agent.rag.local_embed import LocalEmbedder
from cs_agent.rag.retriever import Retriever
from cs_agent.rag.router import Router

mcp = MCPServer("优选商城知识库")

_router = Router(top_n=config.ROUTE_TOP_N) if config.ENABLE_ROUTING else None
_retriever: Optional[Retriever] = None


def _build_retriever() -> Retriever:
    """按配置构建检索器（embedder 提供向量；provider 仅占位，hash 模式兜底）。"""
    embedder = (
        LocalEmbedder(config.LOCAL_EMBED_MODEL) if config.EMBED_BACKEND == "local" else None
    )
    r = Retriever(
        MockProvider(),
        top_k=config.TOP_K,
        threshold=config.SCORE_THRESHOLD,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        embedder=embedder,
        router=_router,
    )
    r.build(load_documents(config.KNOWLEDGE_BASE_DIR))
    return r


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever()
    return _retriever


def _doc(d) -> dict:
    return {"source": d.source, "content": d.content, "score": round(d.score, 4)}


@mcp.tool()
def search(query: str, top_k: int = 3) -> List[dict]:
    """在电商售后知识库中检索，返回最相关的 top_k 个片段（含来源与相关性分数）。"""
    return [_doc(d) for d in _get_retriever().search(query)[:top_k]]


@mcp.tool()
def search_many(queries: List[str]) -> List[dict]:
    """多查询并行检索（多跳问题）：对每个子查询分别召回，合并去重后返回。"""
    return [_doc(d) for d in _get_retriever().search_many(queries)]


@mcp.tool()
def route(query: str) -> List[str]:
    """返回查询命中的知识库分片（文档）列表，按相关度降序；未启用路由时返回空。"""
    if _router is None:
        return []
    return _router.route(query)


@mcp.tool()
def list_shards() -> List[str]:
    """列出知识库的全部分片（文档）。"""
    return sorted(d["source"] for d in load_documents(config.KNOWLEDGE_BASE_DIR))


@mcp.tool()
def health() -> dict:
    """返回知识库检索服务状态（不触发索引构建）。"""
    docs = load_documents(config.KNOWLEDGE_BASE_DIR)
    return {
        "status": "ok",
        "documents": len(docs),
        "embed_backend": config.EMBED_BACKEND,
        "embed_model": config.LOCAL_EMBED_MODEL,
        "routing": config.ENABLE_ROUTING,
    }


if __name__ == "__main__":
    mcp.run()
