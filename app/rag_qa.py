import os
import re
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="/api/rag", tags=["RAG"])

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ambil koleksi dari vectorstore
def get_collection():
    emb = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return Chroma(embedding_function=emb, persist_directory=PERSIST_DIR)

# Filter berbasis pertanyaan
def parse_query(q: str):
    q = q.lower()
    f = {
        "brand": None,
        "bahan_bakar": None,
        "transmisi": None,
        "harga_min": None,
        "harga_max": None
    }

    # Brand
    for b in ["bmw", "toyota", "wuling", "hyundai", "renault", "fortuner", "ayla", "innova"]:
        if b in q:
            f["brand"] = b
            break

    # Bahan bakar
    if "listrik" in q or "electric" in q:
        f["bahan_bakar"] = "listrik"
    elif "hybrid" in q:
        f["bahan_bakar"] = "hybrid"
    elif "diesel" in q:
        f["bahan_bakar"] = "diesel"
    elif "bensin" in q:
        f["bahan_bakar"] = "bensin"

    # Transmisi
    if "matic" in q or "otomatis" in q:
        f["transmisi"] = "matic"
    elif "manual" in q:
        f["transmisi"] = "manual"

    # Harga
    if "di bawah" in q or "<=" in q:
        m = re.search(r"(?:di bawah|<=)\s*rp?\s*([\d\.]+)", q)
        if m:
            f["harga_max"] = int(m.group(1).replace(".", ""))
    elif "lebih dari" in q or "di atas" in q or ">=" in q:
        m = re.search(r"(?:lebih dari|di atas|>=)\s*rp?\s*([\d\.]+)", q)
        if m:
            f["harga_min"] = int(m.group(1).replace(".", ""))

    return f

# Cek apakah metadata dokumen cocok dengan filter
def match_filter(meta: dict, f: dict):
    if not meta: return False
    if f["brand"] and f["brand"] not in str(meta.get("nama_mobil", "")).lower(): return False
    if f["bahan_bakar"] and f["bahan_bakar"] not in str(meta.get("bahan_bakar", "")).lower(): return False
    if f["transmisi"] and f["transmisi"] not in str(meta.get("transmisi", "")).lower(): return False
    if f["harga_min"]:
        harga = int("".join(re.findall(r"\d+", str(meta.get("harga", "")))) or "0")
        if harga < f["harga_min"]: return False
    if f["harga_max"]:
        harga = int("".join(re.findall(r"\d+", str(meta.get("harga", "")))) or "0")
        if harga > f["harga_max"]: return False
    return True

@router.get("")
def rag(pertanyaan: str = Query(...)):
    filters = parse_query(pertanyaan)
    col = get_collection()
    docs = col.similarity_search(pertanyaan, k=30)

    # Apply filter
    results = []
    for d in docs:
        meta = d.metadata or {}
        if match_filter(meta, filters):
            results.append({
                "nama_mobil": meta.get("nama_mobil"),
                "tahun": meta.get("tahun"),
                "harga": meta.get("harga"),
                "usia": meta.get("usia"),
                "bahan_bakar": meta.get("bahan_bakar"),
                "transmisi": meta.get("transmisi"),
                "kapasitas_mesin": meta.get("kapasitas_mesin"),
                "cosine_score": getattr(d, "distance", None)
            })

    if not results:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    # Prioritaskan usia <= 5 tahun
    usia_muda = [r for r in results if r.get("usia") is not None and r["usia"] <= 5]
    if usia_muda:
        final = usia_muda[:5]
    else:
        final = results[:5]

    # Teks jawaban
    lines = []
    for i, r in enumerate(final, 1):
        score_line = ""
        if isinstance(r.get("cosine_score"), float):
            score_line = f"Skor: {r['cosine_score']:.4f}\n"
        lines.append(
            f"{i}. {r['nama_mobil']} ({r['tahun']})\n"
            f"{score_line}"
            f"Harga: {r['harga']}\n"
            f"Usia: {r['usia']} tahun\n"
            f"Bahan Bakar: {r['bahan_bakar']}\n"
            f"Transmisi: {r['transmisi']}\n"
            f"Kapasitas Mesin: {r['kapasitas_mesin']}\n"
        )

    return {"jawaban": "Rekomendasi berdasarkan kemiripan:\n\n" + "\n".join(lines), "rekomendasi": final}
