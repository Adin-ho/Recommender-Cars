import os, re
from pathlib import Path
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PERSIST_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).resolve().parents[1] / "chroma"))

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
VECTOR = Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDINGS)

router = APIRouter()

_BRANDS = [
    "bmw","toyota","honda","suzuki","daihatsu","nissan","mitsubishi",
    "mazda","hyundai","kia","wuling","renault","vw","volkswagen","mercedes","audi"
]

def _parse_filters(q: str):
    ql = q.lower()
    f = {}
    if any(k in ql for k in ["listrik","ev","electric"]): f["bahan_bakar"] = "listrik"
    elif "diesel" in ql:  f["bahan_bakar"] = "diesel"
    elif "bensin" in ql:  f["bahan_bakar"] = "bensin"
    elif "hybrid" in ql:  f["bahan_bakar"] = "hybrid"
    if "matic" in ql:     f["transmisi"] = "matic"
    elif "manual" in ql:  f["transmisi"] = "manual"
    for b in _BRANDS:
        if b in ql:
            f["merek"] = "volkswagen" if b == "vw" else b
            break
    return f

def _parse_numeric(q: str):
    ql = q.lower(); out = {}
    if m := re.search(r"(?:di bawah|max(?:imal)?|<=?)\s*rp?\s*([\d\.]+)", ql):
        out["harga_max"] = int(m.group(1).replace(".", ""))
    if m := re.search(r"tahun\s*(\d{4})\s*ke atas", ql):
        out["tahun_min"] = int(m.group(1))
    if m := re.search(r"(?:<|di ?bawah|kurang dari|max)?\s*(\d{1,2})\s*tahun", ql):
        out["usia_max"] = int(m.group(1))
    return out

def _as_int(x, default=0):
    try: return int(float(x))
    except: return default

def _passes_filters(meta, filters, numeric):
    """Jangan buang dokumen kalau metadatanya kosong — hanya buang jika JELAS tidak cocok."""
    bb = str(meta.get("bahan_bakar","")).lower()
    tr = str(meta.get("transmisi","")).lower()
    mk = str(meta.get("merek","")).lower()
    yr = _as_int(meta.get("tahun",0))
    hg = _as_int(meta.get("harga_angka",0))
    us = _as_int(meta.get("usia",0))

    # filter kategorikal (longgar ke dokumen yang metadata-nya kosong)
    if "bahan_bakar" in filters and bb and filters["bahan_bakar"] not in bb: return False
    if "transmisi"   in filters and tr and filters["transmisi"]   != tr:     return False
    if "merek"       in filters and mk and filters["merek"]       not in mk: return False

    # filter numerik (hanya kalau field ada)
    if "harga_max" in numeric and hg and hg > numeric["harga_max"]: return False
    if "tahun_min" in numeric and yr and yr < numeric["tahun_min"]: return False
    if "usia_max"  in numeric and us and us > numeric["usia_max"]:  return False

    return True

def _format(doc, score):
    m = doc.metadata or {}
    return {
        "nama_mobil": m.get("nama_mobil","-"),
        "tahun": _as_int(m.get("tahun",0)),
        "harga": m.get("harga","-"),
        "harga_angka": _as_int(m.get("harga_angka",0)),
        "usia": _as_int(m.get("usia",0)),
        "bahan_bakar": str(m.get("bahan_bakar","")),
        "transmisi": str(m.get("transmisi","")),
        "kapasitas_mesin": m.get("kapasitas_mesin","-"),
        "cosine_score": round(float(score), 4),
    }

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(query: str = Query(...), k: int = 5, exclude: str = ""):
    filters = _parse_filters(query)
    numeric  = _parse_numeric(query)

    # 1) Ambil kandidat banyak dulu
    raw = VECTOR.similarity_search_with_score(query, k=max(80, 8*k))

    # 2) Terapkan filter dengan “safe mode”
    ex = {x.strip().lower() for x in exclude.split(",") if x.strip()}
    seen = set()
    picked = []
    for doc, score in raw:
        fm = _format(doc, score)
        key = f"{fm['nama_mobil'].lower()}__{fm['tahun']}"
        if fm["nama_mobil"].lower() in ex or key in seen:
            continue
        seen.add(key)
        if _passes_filters(doc.metadata or {}, filters, numeric):
            picked.append(fm)

    # 3) Fallback: kalau kosong, ambil tanpa filter agar user tetap dapat jawaban
    if not picked:
        picked = [_format(doc, score) for doc, score in raw]

    return {"rekomendasi": picked[:max(1, k)]}
