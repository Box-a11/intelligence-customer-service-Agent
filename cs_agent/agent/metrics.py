"""可观测性指标收集。

按请求记录意图 / 延迟 / 检索来源 / 是否澄清 / 是否转人工，供监控大盘查询。

- 累计指标（请求数 / 意图分布 / 来源计数 / 延迟聚合）持久化到 JSON，重启不丢；
- 最近请求明细保留为内存实时窗口（重启清空，等新请求填充）。
`path` 为 None 时退化为纯内存（用于单元测试）。
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


class MetricsStore:
    """线程安全的指标收集器。"""

    MAX_RECENT = 50  # 内存里最多保留最近 N 条请求明细

    def __init__(self, path: Optional[str] = None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._requests = 0
        self._intents: Counter = Counter()
        self._sources: Counter = Counter()
        self._lat_count = 0
        self._lat_sum = 0.0
        self._lat_min: Optional[int] = None
        self._lat_max: Optional[int] = None
        self._recent: List[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text(encoding="utf-8"))
            self._requests = int(d.get("requests", 0))
            self._intents = Counter(d.get("intents", {}))
            self._sources = Counter(d.get("sources", {}))
            lat = d.get("latency", {})
            self._lat_count = int(lat.get("count", 0))
            self._lat_sum = float(lat.get("sum_ms", 0))
            self._lat_min = lat.get("min_ms")
            self._lat_max = lat.get("max_ms")
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "requests": self._requests,
            "intents": dict(self._intents),
            "sources": dict(self._sources),
            "latency": {
                "count": self._lat_count,
                "sum_ms": self._lat_sum,
                "min_ms": self._lat_min,
                "max_ms": self._lat_max,
            },
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
            ms = int(latency_ms)
            self._lat_count += 1
            self._lat_sum += ms
            self._lat_min = ms if self._lat_min is None else min(self._lat_min, ms)
            self._lat_max = ms if self._lat_max is None else max(self._lat_max, ms)
            self._recent.append(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "message": (message or "")[:60],
                    "intent": intent,
                    "latency_ms": ms,
                    "sources": list(sources),
                    "needs_clarification": needs_clarification,
                    "has_ticket": has_ticket,
                }
            )
            if len(self._recent) > self.MAX_RECENT:
                self._recent = self._recent[-self.MAX_RECENT:]
            self._save()

    def snapshot(self) -> dict:
        """返回当前指标的汇总快照。"""
        with self._lock:
            total_sources = sum(self._sources.values())
            return {
                "requests": self._requests,
                "intents": dict(self._intents),
                "latency": {
                    "avg_ms": round(self._lat_sum / self._lat_count) if self._lat_count else 0,
                    "max_ms": self._lat_max or 0,
                    "min_ms": self._lat_min or 0,
                },
                "retrieval": {
                    "total_sources": total_sources,
                    "avg_sources_per_request": round(total_sources / self._requests, 2) if self._requests else 0.0,
                    "top_sources": [
                        {"source": s, "count": c}
                        for s, c in self._sources.most_common(10)
                    ],
                },
                "recent_requests": list(reversed(self._recent)),
            }
