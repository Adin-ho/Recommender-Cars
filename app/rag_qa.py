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
    encode_kwargs={"normalize_embeddings": True},
)
VECTOR = Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDINGS)

router = APIRouter()

_BRANDS = ["bmw","toyota","honda","suzuki","daihatsu","nissan","mitsubishi",
           "mazda","hyundai","kia","wuling","renault","vw","volkswagen","mercedes","audi"]

def _parse_filters(q: str):
    ql = q.lower(); f = {}
    if any(k in ql for k in ["listrik","ev","electric"]): f["bahan_bakar"]="listrik"
    elif "diesel" in ql:  f["bahan_bakar"]="diesel"
    elif "bensin" in ql:  f["bahan_bakar"]="bensin"
    elif "hybrid" in ql:  f["bahan_bakar"]="hybrid"
    if "matic" in ql: f["transmisi"]="matic"
    elif "manual" in ql: f["transmisi"]="manual"
    for b in _BRANDS:
        if b in ql:
            f["merek"] = "volkswagen" if b=="vw" else b
            break
    return f

def _parse_numeric(q: str):
    ql=q.lower(); out={}
    if m:=re.search(r"(?:di bawah|max(?:imal)?|<=?)\s*rp?\s*([\d\.]+)", ql):
        out["harga_max"]=int(m.group(1).replace(".",""))
    if m:=re.search(r"tahun\s*(\d{4})\s*ke atas", ql):
        out["tahun_min"]=int(m.group(1))
    if m:=re.search(r"(?:<|di ?bawah|kurang dari|max)?\s*(\d{1,2})\s*tahun", ql):
        out["usia_max"]=int(m.group(1))
    return out

def _as_int(x, default=0):
    try: return int(float(x))
    except: return default

def _fmt(doc, score):
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
        "merek": str(m.get("merek","")),
        "cosine_score": round(float(score), 4),
    }

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(query: str = Query(...), k: int = 5, exclude: str = ""):
    filters = _parse_filters(query)
    numeric  = _parse_numeric(query)

    raw = VECTOR.similarity_search_with_score(query, k=max(80, 8*k))

    ex = {x.strip().lower() for x in exclude.split(",") if x.strip()}
    seen, picked = set(), []

    for doc, score in raw:
        row = _fmt(doc, score)
        key = f"{row['nama_mobil'].lower()}__{row['tahun']}"
        if row["nama_mobil"].lower() in ex or key in seen:
            continue
        seen.add(key)

        # Kategorikal → longgar (jangan buang kalau metadata kosong)
        if "bahan_bakar" in filters and row["bahan_bakar"] and filters["bahan_bakar"] not in row["bahan_bakar"]:
            continue
        if "transmisi" in filters and row["transmisi"] and filters["transmisi"] != row["transmisi"]:
            continue
        if "merek" in filters and row["merek"] and filters["merek"] not in row["merek"]:
            continue

        # Numerik (hanya jika nilai ada)
        if "harga_max" in numeric and row["harga_angka"] and row["harga_angka"] > numeric["harga_max"]:
            continue
        if "tahun_min" in numeric and row["tahun"] and row["tahun"] < numeric["tahun_min"]:
            continue
        if "usia_max" in numeric and row["usia"] and row["usia"] > numeric["usia_max"]:
            continue

        picked.append(row)

    # Fallback: kalau kosong, tampilkan top-K apa adanya (biar UI tak kosong)
    if not picked:
        picked = [_fmt(doc, s) for doc, s in raw[:k]]

    return {"rekomendasi": picked[:k]}

# Debug: hitung dokumen
@router.get("/debug/chroma_count")
def chroma_count():
    try:
        return {"count": VECTOR._collection.count(), "dir": PERSIST_DIR}
    except Exception as e:
        return {"count": None, "dir": PERSIST_DIR, "error": str(e)}
