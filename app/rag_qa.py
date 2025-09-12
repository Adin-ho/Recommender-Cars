import os
import re
import requests
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="/api/rag", tags=["RAG"])

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _clean_name(nm: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(nm)).strip()

def _parse_query(q: str):
    ql = q.lower()
    parsed = {
        "brand": None,
        "bahan_bakar": None,
        "transmisi": None,
        "harga_max": None
    }

    brands = ["toyota", "daihatsu", "wuling", "bmw", "hyundai", "renault", "innova", "ayla", "fortuner", "mobilio"]
    for b in brands:
        if b in ql:
            parsed["brand"] = b
            break

    if "diesel" in ql:
        parsed["bahan_bakar"] = "diesel"
    elif "bensin" in ql:
        parsed["bahan_bakar"] = "bensin"

    if "matic" in ql or "otomatis" in ql:
        parsed["transmisi"] = "matic"
    elif "manual" in ql:
        parsed["transmisi"] = "manual"

    m = re.search(r"(?:di bawah|<=|maks(?:imal)?)\s*rp?\s*([\d\.]+)", ql)
    if m:
        parsed["harga_max"] = int(m.group(1).replace(".", ""))
    return parsed

def _passes_filter(meta: dict, parsed: dict) -> bool:
    if not meta: return False
    if parsed["brand"] and parsed["brand"] not in str(meta.get("nama_mobil", "")).lower():
        return False
    if parsed["bahan_bakar"] and parsed["bahan_bakar"] not in str(meta.get("bahan_bakar", "")).lower():
        return False
    if parsed["transmisi"] and parsed["transmisi"] not in str(meta.get("transmisi", "")).lower():
        return False
    if parsed["harga_max"]:
        h = str(meta.get("harga", ""))
        digits = re.findall(r"\d+", h)
        harga_num = int("".join(digits)) if digits else 0
        if harga_num > parsed["harga_max"]:
            return False
    return True

def _get_collection():
    emb = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return Chroma(embedding_function=emb, persist_directory=PERSIST_DIR)

@router.get("")
def rag(pertanyaan: str = Query(...), topk: int = 10):
    parsed = _parse_query(pertanyaan)
    col = _get_collection()
    docs = col.similarity_search(pertanyaan, k=30)

    items = []
    for d in docs:
        m = d.metadata or {}
        if not _passes_filter(m, parsed): continue
        items.append({
            "nama_mobil": _clean_name(m.get("nama_mobil")),
            "tahun": m.get("tahun"),
            "harga": m.get("harga"),
            "usia": m.get("usia"),
            "bahan_bakar": m.get("bahan_bakar"),
            "transmisi": m.get("transmisi"),
            "kapasitas_mesin": m.get("kapasitas_mesin"),
            "cosine_score": getattr(d, "distance", None)
        })

    items = sorted(items, key=lambda x: x["cosine_score"] or 0)[:topk]

    if not items:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    lines = []
    for i, r in enumerate(items, 1):
        lines.append(f"{i}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - {r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}")
    return {"jawaban": "Rekomendasi berdasarkan kemiripan:\n\n" + "\n".join(lines), "rekomendasi": items}
