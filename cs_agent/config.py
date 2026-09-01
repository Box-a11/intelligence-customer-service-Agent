"""全局配置：通过环境变量 / .env 文件配置，均可覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base")))
# 会话历史持久化文件路径
HISTORY_PATH = os.getenv("HISTORY_PATH", str(BASE_DIR / "data" / "conversations.json"))
# 用户长期记忆持久化文件路径
MEMORY_PATH = os.getenv("MEMORY_PATH", str(BASE_DIR / "data" / "memory.json"))
# 工单持久化文件路径
TICKET_PATH = os.getenv("TICKET_PATH", str(BASE_DIR / "data" / "tickets.json"))
# 可观测指标累计持久化文件路径（请求数/意图/来源/延迟聚合）
METRICS_PATH = os.getenv("METRICS_PATH", str(BASE_DIR / "data" / "metrics.json"))

# ---- LLM（OpenAI 兼容接口）----
# 未配置 LLM_API_KEY 时自动回退到离线 Mock，保证测试/评测零成本可跑。
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()

# ---- 本地语义 embedding ----
# DeepSeek 不提供 embedding 接口，改用本地 sentence-transformers 中文向量模型
# （BAAI/bge-small-zh-v1.5，已随项目缓存在 HF 缓存目录，首次加载无需联网）。
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5").strip()
# embedding 后端：local=本地 bge 语义向量；hash=离线哈希嵌入（确定性，测试/评测默认）。
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local").strip().lower()

# ---- 检索参数 ----
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
# 检索置信度阈值：最高分低于该阈值时判定「无法回答」并转人工工单。
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))
# ReAct 循环最大轮次（每轮 = 一次思考 + 一次行动）
MAX_REACT_ROUNDS = int(os.getenv("MAX_REACT_ROUNDS", "3"))

# ---- 分库分索引路由 ----
# 是否启用分片路由器：检索时先把查询路由到相关文档分片，只在这些分片上召回。
ENABLE_ROUTING = os.getenv("ENABLE_ROUTING", "1") == "1"
# 路由器最多保留的分片数（命中关键词越多越靠前；无命中回退全量）。
ROUTE_TOP_N = int(os.getenv("ROUTE_TOP_N", "3"))


def use_mock() -> bool:
    """是否使用离线 Mock（未配置 API Key 时为 True）。"""
    return LLM_API_KEY == ""
