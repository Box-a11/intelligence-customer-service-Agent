"""FastAPI 应用：/chat、/sessions、/tickets、/memory、/health，并托管前端页面。"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..agent.history import ConversationStore
from ..agent.memory import MemoryStore
from ..app import build_agent
from ..schemas import ChatRequest, ChatResponse, Ticket

agent, ticket_store, _retriever = build_agent()
history_store = ConversationStore(config.HISTORY_PATH)
memory_store = MemoryStore(config.MEMORY_PATH)

FRONTEND_DIR = config.BASE_DIR / "frontend"

app = FastAPI(
    title="智能客服 Agent",
    version="1.2.0",
    description="基于 RAG + LangGraph 的智能客服：意图识别、多轮澄清、知识库问答、工单转人工、会话历史持久化、用户长期记忆。",
)

# 允许前端从任意来源访问（本地演示：同源访问或直接打开页面均可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _format_memory(mems: list) -> str:
    """把记忆列表格式化为可注入提示词的文本。"""
    if not mems:
        return ""
    lines = []
    for m in mems:
        q = m.get("question", "")
        a = (m.get("answer") or "")[:80]
        lines.append(f"- 用户曾问：「{q}」，当时回答：「{a}」")
    return "\n".join(lines)


@app.get("/health")
def health():
    return {"status": "ok", "llm": agent.provider.name}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    history = history_store.get(req.session_id)
    memory = _format_memory(memory_store.get(req.user_id))
    result = agent.run(req.message, history=history, memory=memory)
    reply = result.get("answer", "")

    # 持久化本轮对话与长期记忆
    history_store.append(req.session_id, "user", req.message)
    history_store.append(req.session_id, "assistant", reply)
    if not result.get("needs_clarification"):
        # 澄清话术不算有效回答，不写入长期记忆
        memory_store.add(req.user_id, req.message, reply[:200])

    return ChatResponse(
        reply=reply,
        intent=result.get("intent"),
        needs_clarification=bool(result.get("needs_clarification")),
        ticket=result.get("ticket"),
        sources=list(result.get("sources", [])),
    )


@app.get("/sessions")
def list_sessions():
    return history_store.list_sessions()


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    msgs = history_store.get(session_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "messages": msgs}


@app.get("/tickets", response_model=list[Ticket])
def list_tickets():
    return ticket_store.list()


@app.get("/tickets/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: str):
    t = ticket_store.get(ticket_id)
    if t is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return t


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    mems = memory_store.get(user_id, limit=50)
    return {"user_id": user_id, "memories": mems}


# ---- 前端页面（最后挂载，避免覆盖 API 路由）----
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
