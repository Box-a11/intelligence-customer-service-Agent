"""LLM 能力抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class LLMProvider(ABC):
    """智能客服所需 LLM 能力的抽象。

    两个实现：
    - OpenAICompatibleProvider：调用 OpenAI 兼容接口（DeepSeek / 通义 / 智谱 / OpenAI 等）
    - MockProvider：离线确定性实现，用于测试与评测（无 Key 即可跑通）
    """

    name: str = "base"

    @abstractmethod
    def classify_intent(self, history: Sequence[dict]) -> str:
        """根据对话历史（[{"role","content"}, ...]）判断最新用户消息的意图。

        返回：qa / consultation / complaint / unclear
        """

    @abstractmethod
    def answer(self, question: str, context: str, memory: str = "") -> str:
        """基于检索到的参考资料回答问题；memory 为该用户的历史记忆（可空）。"""

    @abstractmethod
    def think(self, question: str, context: str = "", memory: str = "", round: int = 0) -> dict:
        """ReAct 思考步骤：决定下一步动作。

        返回 {"action": ..., "content": ...}，action 取值：
        - answer：content 为最终回答
        - retrieve：content 为检索关键词；多跳问题可用换行/分号列出多个子检索词（每行一个）
        - clarify：content 为澄清问题
        - escalate：无法回答，转人工
        """

    @abstractmethod
    def clarify(self, question: str) -> str:
        """针对模糊问题生成澄清提问。"""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """将文本列表编码为向量（L2 归一化）。"""

    def summarize_complaint(self, message: str) -> str:
        """投诉摘要。默认直接返回原消息，真实实现可调用 LLM 生成摘要。"""
        return message
