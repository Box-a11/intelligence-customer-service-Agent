"""pytest 共享 fixture：构建一个基于 MockProvider 的确定性 agent。"""
from __future__ import annotations

import os
import tempfile

# 强制测试离线：.env 若已填真实 Key，测试/评测不应触发真实 API 调用。
# 必须在导入 cs_agent 之前设置，否则 load_dotenv 会读入真实 Key。
os.environ["LLM_API_KEY"] = ""
# 检索也强制回退离线哈希嵌入，避免加载本地 bge 模型（保持测试确定性、快、零外部依赖）。
os.environ["EMBED_BACKEND"] = "hash"
# 会话历史写入临时目录，避免污染项目 data/ 目录的真实对话数据。
os.environ["HISTORY_PATH"] = os.path.join(tempfile.gettempdir(), "cs_agent_test_conversations.json")
# 长期记忆同样写入临时目录
os.environ["MEMORY_PATH"] = os.path.join(tempfile.gettempdir(), "cs_agent_test_memory.json")

import pytest

from cs_agent import config
from cs_agent.agent.graph import CustomerServiceAgent
from cs_agent.agent.ticket import TicketStore
from cs_agent.llm.mock import MockProvider
from cs_agent.rag.loader import load_documents
from cs_agent.rag.retriever import Retriever
from cs_agent.rag.router import Router


@pytest.fixture(scope="session")
def mock_agent() -> dict:
    provider = MockProvider()
    # 与生产一致：启用分库分索引路由器，测试覆盖路由检索路径。
    retriever = Retriever(
        provider,
        top_k=config.TOP_K,
        threshold=config.SCORE_THRESHOLD,
        router=Router(top_n=config.ROUTE_TOP_N),
    )
    retriever.build(load_documents(config.KNOWLEDGE_BASE_DIR))
    store = TicketStore()
    agent = CustomerServiceAgent(provider, retriever, store)
    return {"agent": agent, "retriever": retriever, "store": store, "provider": provider}
