import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import APIRouter, Query
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
PERSIST_DIR = os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma"))

router = APIRouter()

# Reuse sekali saja
EMBEDDINGS = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
VECTOR = Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDINGS)

_BRANDS = [
    "bmw","toyota","honda","suzuki","daihatsu","nissan","mitsubishi",
    "mazda","hyundai","kia","wuling","renault","vw","volkswagen","mercedes","mercedes-benz","audi"
]

def _parse_filters(q: str) -> Dict:
    ql = q.lower()
    f = {}
    # bahan bakar
    if any(k in ql for k in ["listrik","ev","electric"]): f["bahan_bakar"] = "listrik"
    elif "diesel" in ql:  f["bahan_bakar"] = "diesel"
    elif "bensin" in ql:  f["bahan_bakar"] = "bensin"
    elif "hybrid" in ql:  f["bahan_bakar"] = "hybrid"
    # transmisi
    if "matic" in ql:  f["transmisi"] = "matic"
    elif "manual" in ql: f["transmisi"] = "manual"
    # merek
    for b in _BRANDS:
        if b in ql:
            f["merek"] = "volkswagen" if b == "vw" else b
            break
    return f

def _parse_numeric(q: str) -> Dict:
    ql = q.lower()
    out = {}
    # harga (di bawah/max 200 juta / angka penuh)
    if m := re.search(r"(?:di bawah|max(?:imal)?|<=?)\s*rp?\s*([\d\.]+)", ql):
        out["harga_max"] = int(m.group(1).replace(".", ""))
    # tahun ke atas
    if m := re.search(r"tahun\s*(\d{4})\s*ke atas", ql):
        out["tahun_min"] = int(m.group(1))
    # usia max (hanya jika disebut)
    if m := re.search(r"(?:<|di ?bawah|kurang dari|max(?:imal)?)\s*(\d{1,2})\s*tahun", ql):
        out["usia_max"] = int(m.group(1))
    return out

def _valid_int(x, default=0):
    try: return int(float(x))
    except Exception: return default

def _keyword_boost(q: str, meta: Dict) -> float:
    ql = q.lower()
    s = 0.0
    if meta.get("merek") and meta["merek"] in ql: s += 0.6
    if meta.get("bahan_bakar") and meta["bahan_bakar"] in ql: s += 0.4
    if "matic" in ql and meta.get("transmisi") == "matic": s += 0.2
    if "manual" in ql and meta.get("transmisi") == "manual": s += 0.2
    if m := re.search(r"(\d{4})", ql):
        th = int(m.group(1))
        try:
            if int(meta.get("tahun", 0)) >= th: s += 0.2
        except: pass
    return min(s, 1.0)

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(
    query: str = Query(..., description="Pertanyaan kebutuhan mobil"),
    k: int = Query(5, ge=1, le=50),
    exclude: str = Query("", description="Nama mobil yang sudah tampil, pisahkan koma")
):
    ql = query.lower()
    filters = _parse_filters(ql)
    numeric  = _parse_numeric(ql)
    usia_max = numeric.get("usia_max")  # <— hanya dipakai kalau user menyebutkan

    # Ambil kandidat banyak dulu (tanpa filter metadata di Chroma untuk kompatibilitas),
    # lalu saring manual dengan metadata.
    docs_scores = VECTOR.similarity_search_with_score(query, k=max(8*k, 80))

    ex = {x.strip().lower() for x in exclude.split(",") if x.strip()}
    seen = set()
    pool: List[Tuple[dict, float]] = []

    for doc, score in docs_scores:
        m = doc.metadata or {}
        nama = str(m.get("nama_mobil","-")).strip()
        tahun = _valid_int(m.get("tahun",0))
        bb = str(m.get("bahan_bakar","")).lower()
        trans = str(m.get("transmisi","")).lower()
        merek = str(m.get("merek","")).lower()
        harga = _valid_int(m.get("harga_angka",0))
        usia  = _valid_int(m.get("usia",0))

        key = f"{nama.lower()}__{tahun}"
        if not nama or nama.lower() in ex or key in seen:
            continue
        seen.add(key)

        # filter metadata sesuai query (bahan bakar, transmisi, merek)
        if filters.get("bahan_bakar") and filters["bahan_bakar"] not in bb:
            continue
        if filters.get("transmisi") and filters["transmisi"] != trans:
            continue
        if filters.get("merek") and filters["merek"] not in merek:
            continue

        # filter numerik opsional
        if "harga_max" in numeric and harga and harga > numeric["harga_max"]:
            continue
        if "tahun_min" in numeric and tahun and tahun < numeric["tahun_min"]:
            continue
        if usia_max is not None and usia and usia > usia_max:
            continue  # usia HANYA dibatasi jika user menyebutkan

        pool.append(({
            "nama_mobil": nama,
            "tahun": tahun,
            "harga": m.get("harga","-"),
            "harga_angka": harga,
            "usia": usia,
            "bahan_bakar": bb,
            "transmisi": trans,
            "kapasitas_mesin": m.get("kapasitas_mesin","-"),
        }, float(score)))

    if not pool:
        return {"jawaban": "Maaf, tidak ditemukan mobil yang sesuai.", "rekomendasi": []}

    # rerank: gabung cosine + keyword boost kecil
    ranked = []
    for meta, cos in pool:
        kb = _keyword_boost(query, {**meta, "merek": _parse_filters(query).get("merek", meta.get("merek",""))})
        final = 0.85 * float(cos) + 0.15 * kb
        ranked.append((final, meta, cos))
    ranked.sort(key=lambda x: x[0], reverse=True)

    hasil = []
    for _, m, cos in ranked[:k]:
        hasil.append({
            **m,
            "cosine_score": round(float(cos), 4)
        })

    # format jawaban text (opsional)
    out = "Rekomendasi berdasarkan Cosine Similarity:\n\n"
    for i, m in enumerate(hasil, 1):
        out += (
            f"{i}. {m['nama_mobil']} ({m['tahun']})\n"
            f"    Skor: {m['cosine_score']}\n"
            f"    Harga: {m['harga']}\n"
            f"    Usia: {m['usia']} tahun\n"
            f"    Bahan Bakar: {m['bahan_bakar']}\n"
            f"    Transmisi: {m['transmisi']}\n"
            f"    Kapasitas Mesin: {m['kapasitas_mesin']}\n\n"
        )

    return {"jawaban": out, "rekomendasi": hasil}
