"""离线确定性 Provider：基于规则，无需网络与 API Key，用于测试与评测。"""
from __future__ import annotations

from typing import Sequence

from .base import LLMProvider
from ..rag.embeddings import hash_embed


class MockProvider(LLMProvider):
    name = "mock"

    COMPLAINT_KEYS = (
        "投诉", "差评", "太差", "服务差", "垃圾", "坑", "骗", "气死", "不满意",
        "举报", "索赔", "假一", "赔偿", "态度", "太慢", "迟迟", "敷衍",
    )
    QA_KEYS = (
        "退货", "换货", "退款", "退换", "运费", "发票", "积分", "优惠券", "物流",
        "配送", "签收", "闪退", "卡顿", "登录", "密码", "支付", "失败", "不更新",
        "打不开", "冻结", "催件", "故障", "发货", "收货",
    )
    CONSULT_KEYS = (
        "咨询", "想了解", "支持吗", "可以吗", "有没有", "什么时候", "怎么办理",
        "怎么开通", "开通", "分期", "价保", "多少钱", "怎么收费", "包邮",
        "发票类型", "客服电话", "热线", "联系方式", "活动", "新人", "会员", "权益",
    )
    # 已知多跳问题组合：两个主题分属不同知识文档，需分别检索后合并回答。
    MULTIHOP_PAIRS = (
        ("退货", "退款"),   # 03 退换货政策 + 04 退款说明
        ("换货", "退款"),   # 03 + 04
        ("退货", "运费"),   # 03 + 02 配送与物流
        ("优惠券", "退货"),  # 11 大促与优惠 + 03
    )

    def classify_intent(self, history: Sequence[dict]) -> str:
        text = self._last_user(history)
        if any(k in text for k in self.COMPLAINT_KEYS):
            return "complaint"
        if any(k in text for k in self.QA_KEYS):
            return "qa"
        if any(k in text for k in self.CONSULT_KEYS):
            return "consultation"
        if len(text.strip()) < 4:
            return "unclear"
        return "qa"

    def answer(self, question: str, context: str, memory: str = "") -> str:
        context = context.strip()
        if not context:
            return ""
        return f"根据知识库，为您找到以下信息：\n\n{context}"

    def think(self, question: str, context: str = "", memory: str = "", round: int = 0) -> dict:
        context = (context or "").strip()
        if context:
            return {"action": "answer", "content": f"根据知识库，为您找到以下信息：\n\n{context}"}
        if round == 0:
            subs = self._decompose(question)
            if subs:
                # 多跳：用换行列出多个子检索词，图会拆分后并行召回再合并
                return {"action": "retrieve", "content": "\n".join(subs)}
            return {"action": "retrieve", "content": question}
        return {"action": "escalate", "content": question}

    @classmethod
    def _decompose(cls, text: str) -> list[str]:
        """把多跳问题拆成子检索词；单主题返回空列表。"""
        for a, b in cls.MULTIHOP_PAIRS:
            if a in text and b in text:
                return [a, b]
        return []

    def clarify(self, question: str) -> str:
        return ("您的问题有些笼统，可以再具体一点吗？例如您想了解退换货、配送运费、"
                "发票、会员积分、支付方式中的哪一项？")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [hash_embed(t) for t in texts]

    @staticmethod
    def _last_user(history: Sequence[dict]) -> str:
        for m in reversed(list(history)):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""
