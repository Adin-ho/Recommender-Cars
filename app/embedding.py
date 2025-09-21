from __future__ import annotations
import os
import threading
from functools import lru_cache
from typing import List
import numpy as np

# NOTE: model multilingual yang ringan & cocok untuk Indo
# Bisa diganti via ENV: EMBED_MODEL (default di bawah)
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

_model_lock = threading.Lock()
_model = None


def _load_model():
    """Lazy-load sentence-transformers supaya start cepat & thread-safe."""
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(DEFAULT_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Mengembalikan embedding (np.ndarray; shape [N, D]) untuk list teks.
    """
    model = _load_model()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return emb


@lru_cache(maxsize=1024)
def embed_query(q: str) -> np.ndarray:
    """Cache untuk single-query, mengurangi latensi/biaya encode ulang."""
    vec = embed_texts([q])[0]
    return vec
