"""OpenAI 兼容接口 Provider：通过 openai SDK + base_url 接入任意兼容端点。"""
from __future__ import annotations

import json
import re
from typing import Sequence

from .base import LLMProvider
from ..rag.embeddings import hash_embed

SYS_INTENT = (
    "你是电商平台「优选商城」的智能客服。请判断用户最后一句话的意图，"
    "只输出以下四个标签之一：qa（售后/使用类具体问题）、consultation（售前/政策咨询）、"
    "complaint（投诉不满）、unclear（问题模糊需澄清）。只输出标签本身，不要输出任何解释。"
)

SYS_ANSWER = (
    "你是电商平台「优选商城」的智能客服。请严格依据【参考资料】回答用户问题："
    "只使用参考资料中的信息、不编造；若参考资料中没有答案，只回答「我不知道」；"
    "若提供了【用户历史记忆】，可结合用户之前的问题与回答，给出更连贯、个性化的回答；"
    "回答简洁友好、使用中文。"
)

SYS_CLARIFY = (
    "你是电商平台「优选商城」的智能客服。用户的问题比较模糊，"
    "请用一句友好的话向用户澄清，引导用户明确需求（例如具体想了解退换货、配送、发票、会员中的哪方面）。"
)

SYS_THINK = (
    "你是电商平台「优选商城」的智能客服。对用户的问题，请逐步思考并决定下一步动作。"
    "可选动作：answer（直接给出最终回答）、retrieve（需要更多知识库信息）、"
    "clarify（问题信息不足需向用户澄清，content 为澄清问题）、escalate（无法回答，转人工）。"
    "若问题需要结合多个方面的信息才能回答（多跳问题，例如「退货的电器退款多久到账」需要退货流程和退款到账时间两方面），"
    "请在 retrieve 的 content 中用换行分别列出每个方面的检索关键词，每行一个，不要合并成一句话。"
    "严格只输出 JSON：{\"action\": \"answer|retrieve|clarify|escalate\", \"content\": \"...\"}，不要输出任何其他文字。"
)


def _normalize_intent(raw: str) -> str:
    r = (raw or "").strip().lower()
    mapping = {
        "complaint": "complaint", "投诉": "complaint",
        "consultation": "consultation", "咨询": "consultation",
        "unclear": "unclear", "模糊": "unclear",
        "qa": "qa", "问答": "qa",
    }
    for key in ("complaint", "投诉", "consultation", "咨询", "unclear", "模糊", "qa", "问答"):
        if key in r:
            return mapping[key]
    return "unclear"


def _parse_think(raw: str, fallback_question: str) -> dict:
    """解析 ReAct 思考输出；解析失败时兜底（有文本则当回答，否则转人工）。"""
    try:
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            action = str(d.get("action", "")).strip().lower()
            content = str(d.get("content", "")).strip()
            if action in ("answer", "retrieve", "clarify", "escalate") and content:
                return {"action": action, "content": content}
    except Exception:
        pass
    if raw and raw.strip():
        return {"action": "answer", "content": raw.strip()}
    return {"action": "escalate", "content": fallback_question}


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(self, api_key: str, base_url: str, model: str, embed_model: str):
        from openai import OpenAI  # 延迟导入，离线 Mock 模式不强依赖

        self._model = model
        self._embed_model = embed_model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        # 推理类模型（如 deepseek-v4-pro）会先产出 reasoning_content 再产出 content；
        # max_tokens 需覆盖「推理 + 答案」，过小会导致 content 为空。
        return (resp.choices[0].message.content or "").strip()

    def classify_intent(self, history: Sequence[dict]) -> str:
        user = self._format_history(history)
        return _normalize_intent(self._chat(SYS_INTENT, user, max_tokens=1024))

    def answer(self, question: str, context: str, memory: str = "") -> str:
        parts = [f"【参考资料】\n{context or '（无参考资料）'}"]
        if memory:
            parts.append(f"【用户历史记忆】\n{memory}")
        parts.append(f"【用户问题】\n{question}")
        return self._chat(SYS_ANSWER, "\n\n".join(parts))

    def clarify(self, question: str) -> str:
        return self._chat(SYS_CLARIFY, f"【用户问题】\n{question}", max_tokens=1024)

    def think(self, question: str, context: str = "", memory: str = "", round: int = 0) -> dict:
        parts = [
            f"【用户问题】\n{question}",
            f"【已检索到的参考资料】\n{context or '（暂无）'}",
        ]
        if memory:
            parts.append(f"【用户历史记忆】\n{memory}")
        parts.append(f"这是第 {round + 1} 轮思考，请决定下一步动作。")
        raw = self._chat(SYS_THINK, "\n\n".join(parts), max_tokens=1024)
        return _parse_think(raw, question)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._embed_model:
            # 未配置嵌入模型（如 DeepSeek 不提供 embedding 接口），直接本地哈希嵌入
            return [hash_embed(t) for t in texts]
        try:
            resp = self._client.embeddings.create(model=self._embed_model, input=list(texts))
            return [d.embedding for d in resp.data]
        except Exception:
            # 嵌入接口不可用/失败时，回退到本地哈希嵌入
            return [hash_embed(t) for t in texts]

    @staticmethod
    def _format_history(history: Sequence[dict]) -> str:
        lines = []
        for m in history:
            role = "用户" if m.get("role") == "user" else "客服"
            lines.append(f"{role}：{m.get('content', '')}")
        return "\n".join(lines)
