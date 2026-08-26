"""嵌入工具：字符 bigram、词面覆盖度、离线确定性哈希嵌入。"""
from __future__ import annotations

import hashlib
import math

_PUNCT = set("，。！？、；：\"'“”‘’（）【】《》…—·,.!?;:()[]{}<> \t\n\r")
# 常见功能字/虚词（用于词面信号时剥离语气词、疑问词，避免稀释内容词覆盖度）
_STOP_CHARS = set(
    "的了是么吗呢啊吧请这那个一就都很在有我你他想能会说帮给看下里上没和与或"
    "等之其但而则才也就还又再最更太特常全都只仅然因所为所以如果关于经过根据怎样什"
)


def _normalize(text: str) -> str:
    return "".join(c for c in (text or "").lower() if c not in _PUNCT and c not in _STOP_CHARS)


def bigrams(text: str) -> set[str]:
    """剥离标点与停用字后，取字符 bigram 集合（中文双字 / 英文 bigram）。"""
    t = _normalize(text)
    if not t:
        return set()
    return {t[i:i + 2] for i in range(len(t) - 1)} or {t}


def coverage(query: str, doc: str) -> float:
    """查询词面覆盖度：文档覆盖了多少查询中的内容 bigram，取值 [0,1]。"""
    q = bigrams(query)
    if not q:
        return 0.0
    return len(q & bigrams(doc)) / len(q)


def hash_embed(text: str, dim: int = 512) -> list[float]:
    """基于内容 bigram 的 feature hashing 嵌入，L2 归一化。

    确定性、无外部依赖，度量字符/词级重叠相似度，支撑离线模式下的检索与评测。
    有真实 embedding 接口时，由 OpenAICompatibleProvider 优先使用真实向量。
    """
    t = _normalize(text)
    vec = [0.0] * dim
    grams: list[str] = list(bigrams(text))
    if not grams:
        grams = [t or ""]

    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
