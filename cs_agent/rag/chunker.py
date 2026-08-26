"""文档分块：按标题分节，长节再按固定长度切分（带重叠）。"""
from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    text = text.strip()
    lines = text.splitlines()

    # 按标题（# 开头）拆分成「节」
    sections: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.startswith("#"):
            if current:
                sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    chunks: List[str] = []
    for sec in sections:
        block = "\n".join(sec).strip()
        if not block:
            continue
        if len(block) <= chunk_size:
            chunks.append(block)
        else:
            chunks.extend(_split_long(block, chunk_size, overlap))
    return chunks


def _split_long(text: str, chunk_size: int, overlap: int) -> List[str]:
    parts: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        parts.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return parts
