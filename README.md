# 智能客服 Agent

基于 **RAG + LangGraph + FastAPI** 的智能客服，面向电商售后场景：知识库问答、多轮澄清、意图识别（问答/投诉/咨询）、无法回答时自动生成工单转人工。

## 功能特性

- **知识库问答（RAG）**：12 份电商售后文档，混合检索（向量余弦 + 词面覆盖）+ 生成回答。
- **意图识别**：`qa`（问答）/ `consultation`（咨询）/ `complaint`（投诉）/ `unclear`（模糊）。
- **多轮对话**：会话历史持久化，同一 session 内支持追问，重启后仍保留上下文。
- **澄清**：模糊问题主动反问，引导用户补充。
- **工单转人工**：投诉、以及知识库未覆盖的问题，自动生成结构化工单。
- **会话历史**：`GET /sessions`、`GET /sessions/{id}` 查询历史对话记录。
- **长期记忆**：按用户（`user_id`）跨会话记住历史问答，换新会话仍能结合之前上下文作答。
- **ReAct 多轮思考**：`qa`/`consultation` 走有界 ReAct 循环，LLM 每轮决定「回答 / 检索 / 澄清 / 转人工」，复杂与模糊问题可多轮迭代。
- **离线可跑**：未配置 API Key 时自动回退到确定性 Mock，测试/评测零成本复现。

## 目录结构

```
├── knowledge_base/          # 12 份中文电商售后知识文档
├── cs_agent/
│   ├── config.py            # 配置（环境变量 / .env）
│   ├── schemas.py           # Pydantic 数据模型
│   ├── llm/                 # LLM 接入层（OpenAI兼容 + 离线Mock）
│   ├── rag/                 # 加载/切块/嵌入/混合检索
│   ├── agent/               # LangGraph 编排 + 工单存储
│   └── api/main.py          # FastAPI 接口
├── eval/                    # 评测集 + 评测脚本
├── tests/                   # pytest 测试用例
├── docs/                    # 需求文档 + 技术方案
├── reports/                 # 测试报告 + 评测报告
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv/Scripts/pip install -r requirements.txt
# Linux / macOS
.venv/bin/pip install -r requirements.txt
```

### 2. 启动服务（默认离线 Mock）

```bash
# Windows
.venv/Scripts/python -m uvicorn cs_agent.api.main:app --reload
# Linux / macOS
.venv/bin/python -m uvicorn cs_agent.api.main:app --reload
```

访问交互式接口文档：http://127.0.0.1:8000/docs

### 3. 调用接口

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "偏远地区运费怎么算", "session_id": "u1"}'
```

## 接入真实 LLM（可选）

复制 `.env.example` 为 `.env`，填写 `LLM_API_KEY` 等即可切换真实模型：

```dotenv
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro
EMBED_MODEL=
```

支持任意 OpenAI 兼容端点（DeepSeek / 通义千问 / 智谱 / OpenAI 等）。未配置 `LLM_API_KEY` 时自动使用离线 Mock。

## 运行测试与评测

```bash
# 单元/集成测试
.venv/Scripts/python -m pytest

# 评测（生成 reports/评测报告.md）
.venv/Scripts/python eval/run_eval.py
```

## 评测指标

评测集 25 条样本，覆盖 4 类意图与转人工场景：

| 指标 | 说明 |
| ---- | ---- |
| 意图识别准确率 | 预测意图与期望一致的比例 |
| 检索命中率 Hit@TopK | 期望文档出现在 Top-K 召回中的比例 |
| 答案关键信息覆盖 | 回答包含期望关键词的比例 |
| 转人工准确率 | 是否生成工单与期望一致的比例 |

## 更多文档

- 需求文档：`docs/01-需求文档.md`
- 技术方案：`docs/02-技术方案.md`
- 测试报告：`reports/测试报告.md`
