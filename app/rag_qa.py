import os, re
from pathlib import Path
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PERSIST_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).resolve().parents[1] / "chroma"))

EMB = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
VS = Chroma(persist_directory=PERSIST_DIR, embedding_function=EMB)
router = APIRouter()

_BRANDS = ["bmw","toyota","honda","suzuki","daihatsu","nissan","mitsubishi",
           "mazda","hyundai","kia","wuling","renault","vw","volkswagen","mercedes","audi"]

def _parse_filters(q: str):
    ql=q.lower(); f={}
    if any(k in ql for k in ["listrik","ev","electric"]): f["bahan_bakar"]="listrik"
    elif "diesel" in ql:  f["bahan_bakar"]="diesel"
    elif "bensin" in ql:  f["bahan_bakar"]="bensin"
    elif "hybrid" in ql:  f["bahan_bakar"]="hybrid"
    if "matic" in ql: f["transmisi"]="matic"
    elif "manual" in ql: f["transmisi"]="manual"
    for b in _BRANDS:
        if b in ql: f["merek"] = "volkswagen" if b=="vw" else b; break
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

def _i(x, d=0):
    try: return int(float(x))
    except: return d

def _fmt(doc, s):
    m=doc.metadata or {}
    return {
        "nama_mobil": m.get("nama_mobil","-"),
        "tahun": _i(m.get("tahun",0)),
        "harga": m.get("harga","-"),
        "harga_angka": _i(m.get("harga_angka",0)),
        "usia": _i(m.get("usia",0)),
        "bahan_bakar": str(m.get("bahan_bakar","")),
        "transmisi": str(m.get("transmisi","")),
        "kapasitas_mesin": m.get("kapasitas_mesin","-"),
        "merek": str(m.get("merek","")),
        "cosine_score": round(float(s),4),
    }

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(query:str=Query(...), k:int=5, exclude:str=""):
    flt=_parse_filters(query); num=_parse_numeric(query)
    raw = VS.similarity_search_with_score(query, k=max(80,8*k))

    ex = {x.strip().lower() for x in exclude.split(",") if x.strip()}
    seen, picked = set(), []
    for doc, s in raw:
        row=_fmt(doc,s)
        key=f"{row['nama_mobil'].lower()}__{row['tahun']}"
        if row["nama_mobil"].lower() in ex or key in seen: continue
        seen.add(key)

        # kategorikal (longgar)
        if "bahan_bakar" in flt and row["bahan_bakar"] and flt["bahan_bakar"] not in row["bahan_bakar"]: continue
        if "transmisi"   in flt and row["transmisi"]   and flt["transmisi"]   != row["transmisi"]:           continue
        if "merek"       in flt and row["merek"]       and flt["merek"]       not in row["merek"]:           continue
        # numerik (jika ada)
        if "harga_max" in num and row["harga_angka"] and row["harga_angka"]>num["harga_max"]: continue
        if "tahun_min" in num and row["tahun"]       and row["tahun"]      < num["tahun_min"]: continue
        if "usia_max"  in num and row["usia"]        and row["usia"]       > num["usia_max"]:  continue

        picked.append(row)

    if not picked:  # fallback biar tidak pernah kosong
        picked=[_fmt(d,s) for d,s in raw[:k]]

    return {"rekomendasi": picked[:k]}

@router.get("/debug/chroma_count")
def chroma_count():
    try: return {"count": VS._collection.count(), "dir": PERSIST_DIR}
    except Exception as e: return {"count": None, "dir": PERSIST_DIR, "error": str(e)}
