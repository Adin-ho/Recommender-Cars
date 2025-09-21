import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
import chromadb
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from mistralai.client import Mistral
from mistralai.models.embeddings import EmbeddingRequest

# ====== ENV ======
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/data/chroma")).as_posix()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_EMB_MODEL = os.getenv("MISTRAL_EMB_MODEL", "mistral-embed")  # bisa ganti "mistral-embed"
BATCH_SIZE = int(os.getenv("EMB_BATCH", "96"))

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY belum diset di environment variables.")

# ====== Mistral client ======
client = Mistral(api_key=MISTRAL_API_KEY)

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Panggil API embedding Mistral untuk satu batch teks.
    Tenacity dipakai agar robust terhadap network hiccups / 429.
    """
    req = EmbeddingRequest(model=MISTRAL_EMB_MODEL, inputs=texts)
    resp = client.embeddings.create(**req.dict())
    # resp.data: list of objects {index, embedding, object}
    # Urutannya sesuai inputs
    return [d.embedding for d in resp.data]

def _row_to_text(row: pd.Series) -> str:
    # Gabungkan atribut penting dalam format ringkas
    return (
        f"{row.get('nama mobil', row.get('nama_mobil',''))} ({row.get('tahun','')}); "
        f"bahan bakar: {row.get('bahan bakar', row.get('bahan_bakar',''))}; "
        f"transmisi: {row.get('transmisi','')}; "
        f"kapasitas: {row.get('kapasitas mesin', row.get('kapasitas_mesin',''))}; "
        f"harga: {row.get('harga','')}"
    )

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    return df

def build_embeddings_for_texts(texts: List[str]) -> List[List[float]]:
    """Batching + retry untuk seluruh dokumen."""
    out: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        embs = _embed_batch(batch)
        out.extend(embs)
    return out

def upsert_chroma(docs: List[str], metadatas: List[Dict], ids: List[str], embeddings: List[List[float]]):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client_chroma.get_or_create_collection("cars")

    # bersihkan isi lama (aman karena kita tulis ulang)
    if coll.count() > 0:
        existing = coll.get()["ids"]
        if existing:
            coll.delete(ids=existing)

    coll.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)

def rebuild_index(csv_path: str) -> int:
    """
    Bangun ulang index:
    - Baca CSV
    - Susun dokumen & metadata
    - Minta embedding ke Mistral
    - Simpan ke Chroma (persist directory)
    """
    df = load_dataframe(csv_path)
    docs = [_row_to_text(r) for _, r in df.iterrows()]
    metas = df.to_dict(orient="records")
    ids = [f"car-{i}" for i in range(len(df))]

    embs = build_embeddings_for_texts(docs)
    upsert_chroma(docs, metas, ids, embs)
    return len(docs)

def embed_query(text: str) -> List[float]:
    """Untuk query runtime (pencarian) pakai embedding Mistral juga."""
    return _embed_batch([text])[0]
