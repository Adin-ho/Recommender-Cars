# app/embedding.py
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===== Konfigurasi dasar =====
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

# Folder persist (gunakan /data di Zeabur agar persisten)
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "/data/emb"))
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
EMB_FILE = PERSIST_DIR / "embeddings.npz"
META_FILE = PERSIST_DIR / "meta.json"

USE_MISTRAL = os.getenv("USE_MISTRAL_EMB", "1") == "1"
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-embed")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# ====== Helpers ======
def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm

def _cosine_sim(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    # q shape: (d,), M: (N,d)
    qn = q / (np.linalg.norm(q) + 1e-12)
    return (M @ qn)

def _concat_text(row: pd.Series) -> str:
    # Jadikan satu string per mobil agar embedding menangkap konteks
    parts = [
        str(row.get("nama mobil", "")),
        str(row.get("tahun", "")),
        str(row.get("bahan bakar", "")),
        str(row.get("transmisi", "")),
        str(row.get("kapasitas mesin", "")),
        str(row.get("harga", "")),
    ]
    return " | ".join(p.strip() for p in parts if p is not None)

# ====== Embedders ======
class MistralEmbedder:
    def __init__(self, api_key: str, model: str = "mistral-embed") -> None:
        from mistralai import Mistral
        self.client = Mistral(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: List[str], batch_size: int = 96) -> np.ndarray:
        out: List[List[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Mistral embed"):
            chunk = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, inputs=chunk)
            out.extend(e.embedding for e in resp.data)
        return np.asarray(out, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
        return vecs.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        v = self.model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
        return v

def get_embedder():
    if USE_MISTRAL and MISTRAL_API_KEY:
        return MistralEmbedder(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL)
    return LocalEmbedder()

# ====== Public APIs ======
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    # Normalisasi kolom
    df.columns = df.columns.str.strip().str.lower()
    return df

def build_corpus(df: pd.DataFrame) -> List[str]:
    return [_concat_text(row) for _, row in df.iterrows()]

def build_and_persist_embeddings() -> Tuple[np.ndarray, List[int]]:
    df = load_dataset()
    texts = build_corpus(df)
    emb = get_embedder().embed_texts(texts)
    emb = _normalize_rows(emb)

    ids = list(range(len(texts)))
    np.savez_compressed(EMB_FILE, vectors=emb, ids=np.asarray(ids, dtype=np.int32))
    META_FILE.write_text(json.dumps({"count": len(ids)}, ensure_ascii=False))
    return emb, ids

def load_embeddings() -> Tuple[np.ndarray, List[int]]:
    if not EMB_FILE.exists():
        return build_and_persist_embeddings()
    data = np.load(EMB_FILE)
    return data["vectors"], list(data["ids"].tolist())

def query_topk(df: pd.DataFrame, vectors: np.ndarray, query: str, k: int = 5) -> List[Tuple[int, float]]:
    # embed query
    vq = get_embedder().embed_one(query).astype(np.float32)
    scores = _cosine_sim(vq, vectors)  # (N,)
    # top-k
    idx = np.argpartition(-scores, kth=min(k, len(scores)-1))[:max(k,1)]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
