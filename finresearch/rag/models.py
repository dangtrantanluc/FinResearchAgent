"""The two local models: a multilingual encoder and a cross-encoder reranker."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_DIM = 1024
MAX_TOKENS = 512


def _device() -> str:
    """FINRESEARCH_DEVICE=cpu keeps a batch job off the GPU while the app is using it."""
    import torch

    return os.environ.get("FINRESEARCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(module):
    """Halve the weights on the CPU first: loading both models at full precision does not fit in 4 GB."""
    if _device() == "cuda":
        module.half()
        module.to("cuda")
    return module


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = MAX_TOKENS
    return _to_device(model)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder

    return _to_device(CrossEncoder(RERANK_MODEL, device="cpu", max_length=MAX_TOKENS))


def embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Unit-length vectors, so a dot product is the cosine similarity."""
    return embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def rerank(query: str, texts: list[str], batch_size: int = 8) -> np.ndarray:
    return reranker().predict([(query, t) for t in texts], batch_size=batch_size, show_progress_bar=False)
