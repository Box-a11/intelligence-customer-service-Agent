"""组装 agent 的工厂函数。"""
from __future__ import annotations

from typing import Tuple

from . import config
from .agent.graph import CustomerServiceAgent
from .agent.ticket import TicketStore
from .llm.base import LLMProvider
from .llm.mock import MockProvider
from .llm.openai_compat import OpenAICompatibleProvider
from .rag.loader import load_documents
from .rag.local_embed import LocalEmbedder
from .rag.retriever import Retriever
from .rag.router import Router


def get_provider(force_mock: bool = False) -> LLMProvider:
    if force_mock or config.use_mock():
        return MockProvider()
    return OpenAICompatibleProvider(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        embed_model=config.EMBED_MODEL,
    )


def build_agent(force_mock: bool = False) -> Tuple[CustomerServiceAgent, TicketStore, Retriever]:
    provider = get_provider(force_mock)
    # 离线 Mock（测试/评测）走确定性 hash_embed；真实模式且 EMBED_BACKEND=local 时启用本地 bge 模型。
    embedder = None
    if not force_mock and config.EMBED_BACKEND == "local":
        embedder = LocalEmbedder(config.LOCAL_EMBED_MODEL)
    # 分库分索引路由器：检索时先把查询路由到相关文档分片，只在这些分片上召回。
    router = Router(top_n=config.ROUTE_TOP_N) if config.ENABLE_ROUTING else None
    retriever = Retriever(
        provider,
        top_k=config.TOP_K,
        threshold=config.SCORE_THRESHOLD,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        embedder=embedder,
        router=router,
    )
    docs = load_documents(config.KNOWLEDGE_BASE_DIR)
    retriever.build(docs)
    ticket_store = TicketStore()
    agent = CustomerServiceAgent(provider, retriever, ticket_store, max_react_rounds=config.MAX_REACT_ROUNDS)
    return agent, ticket_store, retriever
