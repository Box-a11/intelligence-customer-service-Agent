# CLAUDE.md

## 项目概述

电商售后智能客服 Agent（虚构平台「优选商城」），基于 **RAG + LangGraph + FastAPI**。
功能：知识库问答、意图识别（qa/consultation/complaint/unclear）、多轮澄清、工单转人工、会话历史持久化、用户长期记忆、ReAct 多轮思考、网页前端。

## 技术栈

- **Python 3.13**，虚拟环境在 `.venv/`（根目录的 `venv/` 是 Python 3.6，已废弃，勿用）
- **LangGraph**：流程编排（意图分类 → 路由 → 有界 ReAct 循环）
- **FastAPI + Uvicorn**：服务 + 前端托管
- **NumPy**：向量余弦检索
- **本地 embedding**：`sentence-transformers` + **BAAI/bge-small-zh-v1.5**（中文语义向量，已缓存在 HF 缓存目录；未安装时自动回退离线 `hash_embed`）
- **LLM**：OpenAI 兼容接口 → **DeepSeek `deepseek-v4-pro`**（`.env` 已填真实 Key）
- **前端**：`frontend/` 原生 HTML/CSS/JS，无构建工具

## 目录结构

```
knowledge_base/   # 12 份电商售后文档（RAG 知识库）
cs_agent/
  config.py       # 配置（读 .env）
  schemas.py      # Pydantic 模型（ChatRequest 含 message/session_id/user_id）
  llm/            # base(接口) / openai_compat / mock（可插拔，无 Key 回退 Mock）
  rag/            # loader / chunker / embeddings(hash_embed) / local_embed(bge) / router(分片路由) / retriever(混合检索)
  agent/          # graph.py(ReAct) / ticket.py / history.py / memory.py / metrics.py(可观测指标)
  app.py          # build_agent() 工厂
  api/main.py     # FastAPI 路由 + 前端托管
frontend/         # index.html + styles.css + app.js + dashboard(监控大盘)
eval/             # eval_set.jsonl + run_eval.py
tests/            # 43 个 pytest 用例
docs/             # 01-需求文档 / 02-技术方案
reports/          # 测试报告 / 评测报告
data/             # conversations.json + memory.json（运行时生成，已 gitignore）
mcp_server.py     # RAG MCP Server（search/route 等检索工具，供外部 Agent 调用）
.mcp.json         # MCP Server 注册配置（rag-kb）
.claude/skills/   # 项目 skill：eval(评测) / sync-docs(文档同步)
```

## 常用命令

```bash
# 启动服务（前端在 http://127.0.0.1:8000/）
.venv/Scripts/python -m uvicorn cs_agent.api.main:app --host 127.0.0.1 --port 8000

# 测试（强制离线 Mock，见下方注意事项）
.venv/Scripts/python -m pytest

# 评测（默认离线 Mock；--real 用真实模型）
.venv/Scripts/python eval/run_eval.py
```

## API 接口

- `POST /chat` — `{message, session_id, user_id}` → `{reply, intent, needs_clarification, ticket, sources}`
- `GET /sessions`、`GET /sessions/{id}` — 会话历史
- `GET /tickets`、`GET /tickets/{id}` — 工单
- `GET /memory/{user_id}` — 用户长期记忆
- `GET /health`
- `GET /metrics` — 监控大盘聚合数据（系统信息 + 实时指标 + 持久化计数）
- `GET /dashboard` — 监控大盘前端页面（`http://127.0.0.1:8000/dashboard`）

## 核心设计

1. **RAG 混合检索**：`score = 0.3*余弦 + 0.7*词面覆盖`；bigram 前先剥离停用字（的/了/是/怎么/请问…）。稠密向量默认用本地 bge（`EMBED_BACKEND=local`），离线/未安装时回退 `hash_embed`（feature hashing）。**分库分索引**：按文档分片建独立索引，`Router`（关键词路由）先定位相关分片再检索（`ENABLE_ROUTING`/`ROUTE_TOP_N`），无命中回退全量。
2. **ReAct 循环**（qa/consultation）：`provider.think()` 返回 `{action, content}`，action ∈ answer/retrieve/clarify/escalate，`MAX_REACT_ROUNDS=3`（config）。**多跳问题**：`retrieve` 的 content 可用换行/分号列多个子检索词，`retriever.search_many()` 并行召回去重合并后综合回答。
3. **长期记忆**：`MemoryStore` 按 `user_id` 存历史问答（上限 50 条），每轮注入 `【用户历史记忆】`，澄清话术不记。
4. **图无状态**：`agent.run(message, history, memory)` 显式传历史/记忆，不依赖 checkpointer，跨重启保持上下文。
5. **LLM 可插拔**：无 `LLM_API_KEY` → Mock（确定性）；有 → OpenAI 兼容。

## 重要注意事项（坑，务必读）

1. **`deepseek-v4-pro` 是推理模型**：先输出 `reasoning_content` 再输出 `content`，`max_tokens` 必须给足（openai_compat.py 已设 1024/2048），过小会 `content` 为空。
2. **DeepSeek 无 embedding 接口**：检索稠密向量改走本地 bge（`EMBED_BACKEND=local` + `LOCAL_EMBED_MODEL=BAAI/bge-small-zh-v1.5`）；`LocalEmbedder` 懒加载、加载/推理失败自动回退 `hash_embed`，`EMBED_MODEL` 留空即对 OpenAI 兼容端点回退。
3. **测试/评测强制离线**：`tests/conftest.py` 设 `LLM_API_KEY=""`、`EMBED_BACKEND="hash"`、`HISTORY_PATH`/`MEMORY_PATH` 指向临时目录，避免触发真实 API、加载 bge 模型和污染 `data/`。
4. **`.env` 含真实 Key**（已 gitignore；本项目不是 git 仓库）。
5. **真实模型的意图有软边界**：qa/consultation 判断与 Mock 关键词规则可能不一致（如「退款多久到账」可能判 consultation），但走同一 RAG 路径，不影响效果。
6. **推理模型较慢**：每轮 2~15 秒属正常。

## 验证状态

- 测试：**43 passed**（意图/RAG/ReAct/工单/历史/记忆/接口/路由/指标全覆盖）
- 评测：意图准确率/检索命中/答案覆盖/转人工 **4 项均 100%**（离线 Mock）
