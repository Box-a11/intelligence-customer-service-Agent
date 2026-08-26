"""LangGraph 编排：意图分类 → 路由 → ReAct 循环（思考/检索/澄清/转人工）。

qa / consultation 走有界 ReAct 循环：LLM 每轮决定下一步（answer / retrieve / clarify / escalate），
支持复杂问题的多轮检索与模糊问题的多轮澄清，超过最大轮次后强制收敛。
"""
from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..llm.base import LLMProvider
from ..rag.retriever import Retriever
from .ticket import TicketStore


class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    intent: str
    clarify_question: str
    answer: str
    needs_clarification: bool
    ticket: Optional[dict]
    sources: List[str]
    memory: str
    # ReAct 循环状态
    react_context: str
    react_round: int
    react_action: str
    react_content: str


def _last_user(state: Dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# 子查询分隔符：多跳分解时，LLM 在 retrieve 的 content 中用换行/分号列出多个检索关键词。
_SPLIT_RE = re.compile(r"[\n；;]+")


def _split_queries(content: str) -> List[str]:
    """把 retrieve 的 content 拆成若干子查询（多跳分解）。

    单跳问题 content 即单个查询；多跳问题 content 含多个换行/分号分隔的检索关键词，
    拆开后并行召回、合并上下文。
    """
    parts = [p.strip() for p in _SPLIT_RE.split(content or "")]
    seen: List[str] = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    return seen or ([content.strip()] if content and content.strip() else [])


class CustomerServiceAgent:
    def __init__(
        self,
        provider: LLMProvider,
        retriever: Retriever,
        ticket_store: TicketStore,
        max_react_rounds: int = 3,
    ):
        self.provider = provider
        self.retriever = retriever
        self.ticket_store = ticket_store
        self.max_react_rounds = max_react_rounds
        self.graph = self._build()

    # ---- 图构建 ----
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("classify_intent", self._node_classify)
        g.add_node("react_think", self._node_react_think)
        g.add_node("react_retrieve", self._node_react_retrieve)
        g.add_node("react_answer", self._node_react_answer)
        g.add_node("react_finalize", self._node_react_finalize)
        g.add_node("react_clarify", self._node_react_clarify)
        g.add_node("clarify", self._node_clarify)
        g.add_node("handle_complaint", self._node_complaint)
        g.add_node("escalate_ticket", self._node_escalate)

        g.add_edge(START, "classify_intent")
        g.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {"complaint": "handle_complaint", "unclear": "clarify", "react": "react_think"},
        )
        g.add_conditional_edges(
            "react_think",
            self._route_react,
            {
                "answer": "react_answer",
                "retrieve": "react_retrieve",
                "clarify": "react_clarify",
                "escalate": "escalate_ticket",
            },
        )
        g.add_conditional_edges(
            "react_retrieve",
            self._route_after_retrieve,
            {"continue": "react_think", "finalize": "react_finalize", "escalate": "escalate_ticket"},
        )

        g.add_edge("react_answer", END)
        g.add_edge("react_finalize", END)
        g.add_edge("react_clarify", END)
        g.add_edge("clarify", END)
        g.add_edge("handle_complaint", END)
        g.add_edge("escalate_ticket", END)

        return g.compile()

    # ---- 节点 ----
    def _node_classify(self, state: AgentState) -> dict:
        intent = self.provider.classify_intent(state.get("messages", []))
        return {"intent": intent}

    def _route_intent(self, state: AgentState) -> str:
        intent = state.get("intent")
        if intent == "complaint":
            return "complaint"
        if intent == "unclear":
            return "unclear"
        return "react"

    def _node_react_think(self, state: AgentState) -> dict:
        q = _last_user(state)
        context = state.get("react_context", "") or ""
        round_ = state.get("react_round", 0) or 0
        memory = state.get("memory", "") or ""
        d = self.provider.think(q, context, memory, round_)
        return {"react_action": d["action"], "react_content": d.get("content", "")}

    def _route_react(self, state: AgentState) -> str:
        return state.get("react_action", "escalate")

    def _node_react_retrieve(self, state: AgentState) -> dict:
        query = state.get("react_content", "") or _last_user(state)
        # 多跳问题：content 可能含多个换行/分号分隔的子查询，拆分后并行召回、合并。
        queries = _split_queries(query)
        docs = [d for d in self.retriever.search_many(queries) if d.score >= self.retriever.threshold]
        context = state.get("react_context", "") or ""
        sources = list(state.get("sources", []) or [])
        new_parts = []
        for d in docs:
            if d.source not in sources:
                sources.append(d.source)
            new_parts.append(d.content)
        if new_parts:
            context = (context + "\n\n" + "\n\n".join(new_parts)).strip()
        round_ = (state.get("react_round", 0) or 0) + 1
        return {"react_context": context, "sources": sources, "react_round": round_}

    def _route_after_retrieve(self, state: AgentState) -> str:
        round_ = state.get("react_round", 0) or 0
        if round_ >= self.max_react_rounds:
            # 达到最大轮次：有上下文则强制回答，否则转人工
            return "finalize" if (state.get("react_context", "") or "").strip() else "escalate"
        return "continue"

    def _node_react_answer(self, state: AgentState) -> dict:
        return {
            "answer": state.get("react_content", ""),
            "needs_clarification": False,
            "ticket": None,
            "sources": list(state.get("sources", []) or []),
        }

    def _node_react_finalize(self, state: AgentState) -> dict:
        q = _last_user(state)
        context = state.get("react_context", "") or ""
        memory = state.get("memory", "") or ""
        answer = self.provider.answer(q, context, memory)
        return {
            "answer": answer,
            "needs_clarification": False,
            "ticket": None,
            "sources": list(state.get("sources", []) or []),
        }

    def _node_react_clarify(self, state: AgentState) -> dict:
        cq = state.get("react_content", "") or "您的问题有些笼统，可以再具体一点吗？"
        return {"answer": cq, "clarify_question": cq, "needs_clarification": True}

    def _node_clarify(self, state: AgentState) -> dict:
        q = _last_user(state)
        cq = self.provider.clarify(q)
        return {"clarify_question": cq, "answer": cq, "needs_clarification": True}

    def _node_complaint(self, state: AgentState) -> dict:
        q = _last_user(state)
        summary = self.provider.summarize_complaint(q)
        ticket = self.ticket_store.create("complaint", q, summary)
        reply = (
            "非常抱歉给您带来不好的体验，已为您登记投诉工单，专属客服会尽快联系您处理。"
            f"工单号：{ticket.id}"
        )
        return {"ticket": ticket.model_dump(), "answer": reply, "needs_clarification": False}

    def _node_escalate(self, state: AgentState) -> dict:
        q = _last_user(state)
        intent = state.get("intent", "qa")
        ticket = self.ticket_store.create(intent, q, q)
        reply = (
            "抱歉，我暂时无法准确解答您的问题，已为您生成工单转人工处理。"
            f"工单号：{ticket.id}，专属客服会尽快联系您。"
        )
        return {"ticket": ticket.model_dump(), "answer": reply, "needs_clarification": False}

    # ---- 对外 ----
    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
        memory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行一轮对话。

        history: 该会话在此之前的对话历史（[{"role","content"}...]，不含本条消息），
                 为空视为新会话。由 API 层从持久化存储加载后传入。
        memory: 该用户的长期记忆（跨会话的格式化文本，可空）。
        """
        msgs = list(history or []) + [{"role": "user", "content": message}]
        return self.graph.invoke({"messages": msgs, "memory": memory or ""})
