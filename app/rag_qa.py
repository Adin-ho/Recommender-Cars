# app/rag_qa.py
import os, re
from pathlib import Path
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Non-GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", Path(__file__).resolve().parents[1] / "chroma"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cars")

EMB = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# 1 global vector store (hemat memori)
VS = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(PERSIST_DIR),
    embedding_function=EMB,
)

router = APIRouter()

_BRANDS = [
    "bmw","toyota","honda","suzuki","daihatsu","nissan","mitsubishi",
    "mazda","hyundai","kia","wuling","renault","vw","volkswagen","mercedes","audi"
]

def _i(x, d=0):
    try:
        return int(float(x))
    except Exception:
        return d

def _parse_query(q: str):
    ql = q.lower()
    # harga target (200 juta) atau angka penuh
    harga_target = None
    if m := re.search(r"(\d{2,4})\s*juta", ql):
        harga_target = int(m.group(1)) * 1_000_000
    elif m := re.search(r"(\d{9,12})", ql.replace(".","")):
        harga_target = int(m.group(1))

    usia_max = None
    if m := re.search(r"(?:<|di ?bawah|kurang dari|max(?:imal)?)\s*(\d{1,2})\s*tahun", ql):
        usia_max = int(m.group(1))

    fuel = None
    for bb in ["listrik","electric","diesel","bensin","hybrid"]:
        if bb in ql:
            fuel = "listrik" if bb == "electric" else bb
            break

    trans = None
    if "matic" in ql and "manual" not in ql:
        trans = "matic"
    elif "manual" in ql and "matic" not in ql:
        trans = "manual"

    brand = None
    for b in _BRANDS:
        if b in ql:
            brand = "volkswagen" if b == "vw" else b
            break

    return harga_target, usia_max, fuel, trans, brand

def _fmt(doc, s):
    m = doc.metadata or {}
    return {
        "nama_mobil": m.get("nama_mobil", "-"),
        "tahun": _i(m.get("tahun", 0)),
        "harga": m.get("harga", "-"),
        "harga_angka": _i(m.get("harga_angka", 0)),
        "usia": _i(m.get("usia", 0)),
        "bahan_bakar": str(m.get("bahan_bakar", "")),
        "transmisi": str(m.get("transmisi", "")),
        "kapasitas_mesin": m.get("kapasitas_mesin", "-"),
        "merek": str(m.get("merek", "")),
        "cosine_score": round(float(s), 4),
    }

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(
    query: str = Query(..., description="Contoh: 'rekomendasi mobil diesel 300 juta'"),
    k: int = Query(5, ge=1, le=50),
    exclude: str = ""
):
    harga_target, usia_max, fuel, trans, brand = _parse_query(query)

    # Makin besar k internal → peluang match relevan lebih besar
    raw = VS.similarity_search_with_score(query, k=max(80, 8*k))

    ex = {x.strip().lower() for x in exclude.split(",") if x.strip()}
    seen, utama, cad1, cad2 = set(), [], [], []
    tol = 0.18
    hmin = int(harga_target*(1-tol)) if harga_target else 0
    hmax = int(harga_target*(1+tol)) if harga_target else 10**12

    for doc, score in raw:
        m = doc.metadata or {}
        nama = str(m.get("nama_mobil", "-"))
        key  = f"{nama.lower().strip()}__{m.get('tahun','-')}"
        if nama.lower() in ex or key in seen:
            continue
        seen.add(key)

        harga = _i(m.get("harga_angka", 0))
        usia  = _i(m.get("usia", 0))
        ok = True

        # filter numerik
        if harga_target is not None and not (hmin <= harga <= hmax): ok = False
        if usia_max is not None and usia > usia_max: ok = False

        # filter kategori
        if fuel and str(m.get("bahan_bakar","")) != fuel: ok = False
        if trans and str(m.get("transmisi","")) != trans: ok = False
        if brand and str(m.get("merek","")) != brand: ok = False

        item = _fmt(doc, score)
        if ok:
            utama.append(item)
        elif harga_target is not None:
            cad1.append(item)
        else:
            cad2.append(item)

        if len(utama) >= k:
            break

    hasil = utama or cad1[:k] or cad2[:k]
    return {"rekomendasi": hasil}
