"""本地语义嵌入：加载 BAAI/bge-small-zh-v1.5（sentence-transformers 格式）。

与 `embeddings.hash_embed` 的 feature-hashing 兜底不同，这里给出真正的稠密语义向量，
对中文短句的语义区分能力更强。设计原则：

- 懒加载：首次 `embed()` 才导入并加载模型，避免拖慢服务启动与离线测试。
- 可回退：未安装 sentence-transformers / 模型缺失 / 加载或推理失败时，静默回退到
  `hash_embed`，保证任何环境下检索链路都能跑通（与 provider.embed 的兜底策略一致）。
"""
from __future__ import annotations

from typing import Sequence

from .embeddings import hash_embed


class LocalEmbedder:
    """基于 sentence-transformers 的本地中文向量模型封装。

    embed() 返回与 hash_embed 同规格的结果：list[list[float]]，每个向量 L2 归一化，
    可直接被 `Retriever` 当作 provider.embed() 的替代使用。
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._failed = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._failed:
            return
        try:
            import os

            # 模型已随项目缓存在 HF 缓存目录，强制离线加载：否则 huggingface_hub 会为
            # 不存在的 PEFT 文件发起网络请求，网络不可达时触发多次重试拖慢启动。
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            from sentence_transformers import SentenceTransformer  # 延迟导入

            self._model = SentenceTransformer(self._model_name, device=self._device)
        except Exception:
            self._model = None
            self._failed = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        if self._model is None:
            return [hash_embed(t) for t in texts]
        try:
            vectors = self._model.encode(list(texts), normalize_embeddings=True)
            return [v.tolist() for v in vectors]
        except Exception:
            self._model = None
            self._failed = True
            return [hash_embed(t) for t in texts]
