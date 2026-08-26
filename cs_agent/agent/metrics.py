"""可观测性指标收集（内存实现）。

按请求记录意图 / 延迟 / 检索来源 / 是否澄清 / 是否转人工，供监控大盘查询。
内存实现：指标是「实时可观测」数据，重启即清零；会话 / 工单 / 记忆等持久化数据由
对应 Store 提供，大盘在 `/metrics` 里把两者合并。
"""
from __future__ import annotations

import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List


class MetricsStore:
    """线程安全的内存指标收集器。"""

    MAX_RECENT = 50  # 最多保留最近 N 条请求明细

    def __init__(self):
        self._lock = threading.Lock()
        self._requests = 0
        self._intents: Counter = Counter()
        self._sources: Counter = Counter()
        self._latency_ms: List[int] = []
        self._recent: List[dict] = []

    def record(
        self,
        *,
        intent: str,
        latency_ms: float,
        sources: List[str],
        needs_clarification: bool,
        has_ticket: bool,
        message: str,
    ) -> None:
        """记录一轮对话请求的观测指标。"""
        with self._lock:
            self._requests += 1
            if intent:
                self._intents[intent] += 1
            for s in sources:
                self._sources[s] += 1
            self._latency_ms.append(int(latency_ms))
            self._recent.append(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "message": (message or "")[:60],
                    "intent": intent,
                    "latency_ms": int(latency_ms),
                    "sources": list(sources),
                    "needs_clarification": needs_clarification,
                    "has_ticket": has_ticket,
                }
            )
            if len(self._recent) > self.MAX_RECENT:
                self._recent = self._recent[-self.MAX_RECENT:]

    def snapshot(self) -> dict:
        """返回当前指标的汇总快照。"""
        with self._lock:
            lat = self._latency_ms
            n = self._requests
            total_sources = sum(self._sources.values())
            return {
                "requests": n,
                "intents": dict(self._intents),
                "latency": {
                    "avg_ms": round(sum(lat) / len(lat)) if lat else 0,
                    "max_ms": max(lat) if lat else 0,
                    "min_ms": min(lat) if lat else 0,
                },
                "retrieval": {
                    "total_sources": total_sources,
                    "avg_sources_per_request": round(total_sources / n, 2) if n else 0.0,
                    "top_sources": [
                        {"source": s, "count": c}
                        for s, c in self._sources.most_common(10)
                    ],
                },
                "recent_requests": list(reversed(self._recent)),
            }
