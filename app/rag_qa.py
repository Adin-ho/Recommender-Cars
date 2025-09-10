import os
import requests
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="/api/rag", tags=["RAG"])

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Mistral/OpenRouter (opsional)
USE_MISTRAL = bool(os.getenv("MISTRAL_API_KEY"))
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_BASE = os.getenv("MISTRAL_BASE", "https://api.mistral.ai")

USE_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

def _mistral_chat(prompt: str, context: str) -> str | None:
    """Pakai hosted API jika kunci tersedia. Return None jika tak tersedia."""
    if USE_MISTRAL:
        try:
            r = requests.post(
                f"{MISTRAL_BASE}/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
                json={
                    "model": MISTRAL_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a concise assistant. Use the given context to answer."},
                        {"role": "user", "content": f"Question:\n{prompt}\n\nContext:\n{context}\n\nAnswer briefly."}
                    ],
                    "temperature": 0.2
                },
                timeout=60,
            )
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("[MISTRAL ERR]", e)

    if USE_OPENROUTER:
        try:
            r = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                         "HTTP-Referer": "https://zeabur.app", "X-Title": "car-recommender"},
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a concise assistant. Use the given context to answer."},
                        {"role": "user", "content": f"Question:\n{prompt}\n\nContext:\n{context}\n\nAnswer briefly."}
                    ],
                    "temperature": 0.2
                },
                timeout=60,
            )
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("[OPENROUTER ERR]", e)
    return None

def _get_collection():
    emb = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return Chroma(embedding_function=emb, persist_directory=PERSIST_DIR)

@router.get("", summary="RAG recommend")
def rag(pertanyaan: str = Query(..., description="Pertanyaan user"), topk: int = 5):
    col = _get_collection()
    docs = col.similarity_search(pertanyaan, k=topk)

    if not docs:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    # bentuk list
    hasil = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        hasil.append({
            "rank": i,
            "nama_mobil": m.get("nama_mobil"),
            "tahun": m.get("tahun"),
            "harga": m.get("harga"),
            "usia": m.get("usia"),
            "bahan_bakar": m.get("bahan_bakar"),
            "transmisi": m.get("transmisi"),
            "kapasitas_mesin": m.get("kapasitas_mesin"),
            "cosine_score": getattr(d, "distance", None)  # tergantung versi chroma/langchain
        })

    # ringkasan
    plain = "\n".join(
        f"{i}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - {r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}"
        for i, r in enumerate(hasil, 1)
    )
    jawaban = f"Rekomendasi berdasarkan kemiripan:\n\n{plain}"

    # jika ada API Mistral/OpenRouter, buat penjelasan singkat
    llm_ans = _mistral_chat(pertanyaan, plain)
    if llm_ans:
        jawaban = llm_ans

    return {"jawaban": jawaban, "rekomendasi": hasil}
