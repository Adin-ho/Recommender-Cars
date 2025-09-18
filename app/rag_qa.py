import os
import re
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="/api/rag", tags=["RAG"])

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PREFER_MAX_USIA = int(os.getenv("PREFER_MAX_USIA", "5"))

FUEL_KEYWORDS = {
    "listrik": ["listrik", "electric", "ev"],
    "hybrid": ["hybrid", "hev", "phev", "plugin"],
    "diesel": ["diesel"],
    "bensin": ["bensin", "gasoline", "pertalite", "pertamax"]
}
BRANDS = ["bmw","toyota","daihatsu","wuling","hyundai","renault","honda","suzuki","ford","mitsubishi","innova","fortuner","ayla","pajero","mobilio"]

def get_collection():
    emb = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return Chroma(embedding_function=emb, persist_directory=PERSIST_DIR)

def _parse_int(v: str) -> int:
    if not v: return 0
    v = v.lower().replace("juta", "000000").replace("jt", "000000")
    return int("".join(re.findall(r"\d+", v)) or "0")

def parse_query(q: str):
    q = q.lower()
    f = {"brand": None, "fuel": None, "transmisi": None, "harga_min": None, "harga_max": None}
    for b in BRANDS:
        if b in q: f["brand"]=b; break
    for key, keys in FUEL_KEYWORDS.items():
        if any(k in q for k in keys): f["fuel"]=key; break
    if "matic" in q or "otomatis" in q: f["transmisi"]="matic"
    elif "manual" in q: f["transmisi"]="manual"
    m = re.search(r"(?:di bawah|<=|maks(?:imal)?|max)\s*([^\s]+(?:\s*(?:jt|juta))?)", q)
    if m: f["harga_max"]=_parse_int(m.group(1))
    m = re.search(r"(?:di atas|lebih dari|>=|min(?:imal)?)\s*([^\s]+(?:\s*(?:jt|juta))?)", q)
    if m: f["harga_min"]=_parse_int(m.group(1))
    return f

def _fuel_match(meta_val: str, want: str) -> bool:
    if not want: return True
    s = str(meta_val).lower()
    return any(k in s for k in FUEL_KEYWORDS.get(want,[want]))

def match_filter(meta: dict, f: dict):
    if not meta: return False
    if f["brand"] and f["brand"] not in str(meta.get("nama_mobil","")).lower(): return False
    if not _fuel_match(meta.get("bahan_bakar",""), f["fuel"]): return False
    if f["transmisi"] and f["transmisi"] not in str(meta.get("transmisi","")).lower(): return False
    if f["harga_min"]:
        harga = int("".join(re.findall(r"\d+", str(meta.get("harga","")))) or "0")
        if harga < f["harga_min"]: return False
    if f["harga_max"]:
        harga = int("".join(re.findall(r"\d+", str(meta.get("harga","")))) or "0")
        if harga > f["harga_max"]: return False
    return True

@router.get("")
def rag(pertanyaan: str = Query(...), topk: int = Query(5, ge=1, le=50)):
    f = parse_query(pertanyaan)
    col = get_collection()
    docs = col.similarity_search(pertanyaan, k=30)

    results = []
    for d in docs:
        m = d.metadata or {}
        if match_filter(m, f):
            results.append({
                "nama_mobil": m.get("nama_mobil"),
                "tahun": m.get("tahun"),
                "harga": m.get("harga"),
                "usia": m.get("usia"),
                "bahan_bakar": m.get("bahan_bakar"),
                "transmisi": m.get("transmisi"),
                "kapasitas_mesin": m.get("kapasitas_mesin"),
                "cosine_score": getattr(d, "distance", None)
            })

    if not results:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    muda = [r for r in results if r.get("usia") is not None and r["usia"] <= PREFER_MAX_USIA]
    final = (muda if muda else results)[:topk]

    lines = []
    for i, r in enumerate(final, 1):
        score = r.get("cosine_score")
        score_line = f"Skor: {score:.4f}\n" if isinstance(score, float) else ""
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
