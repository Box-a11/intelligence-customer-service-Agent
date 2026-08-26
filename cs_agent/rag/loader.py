"""知识库文档加载。"""
from __future__ import annotations

from pathlib import Path
from typing import List


def load_documents(kb_dir) -> List[dict]:
    """加载知识库目录下所有 .md 文档，返回 [{"source": 文件名, "content": 全文}]。"""
    kb_dir = Path(kb_dir)
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.stem, "content": text})
    return docs
