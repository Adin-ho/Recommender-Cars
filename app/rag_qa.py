import os, re
from pathlib import Path
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PERSIST_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).resolve().parents[1] / "chroma"))

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
VECTOR = Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDINGS)
router = APIRouter()

_BRANDS = [
    "bmw","toyota","honda","suzuki","renault","wuling","mitsubishi","daihatsu",
    "nissan","mazda","hyundai","kia","vw","volkswagen","mercedes","audi"
]

def _parse_filters(q: str):
    ql = q.lower()
    f = {}
    if any(k in ql for k in ["listrik","ev","electric"]): f["bahan_bakar"]="listrik"
    elif "diesel" in ql: f["bahan_bakar"]="diesel"
    elif "bensin" in ql: f["bahan_bakar"]="bensin"
    elif "hybrid" in ql: f["bahan_bakar"]="hybrid"
    if "matic" in ql: f["transmisi"]="matic"
    elif "manual" in ql: f["transmisi"]="manual"
    for b in _BRANDS:
        if b in ql: f["merek"] = "volkswagen" if b=="vw" else b
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

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(query:str=Query(...), k:int=5, exclude:str=""):
    docs_scores = VECTOR.similarity_search_with_score(query, k=80)
    filters=_parse_filters(query)
    numeric=_parse_numeric(query)

    hasil=[]; seen=set()
    for doc,score in docs_scores:
        m=doc.metadata or {}
        nama=m.get("nama_mobil","-"); tahun=_as_int(m.get("tahun",0))
        harga=_as_int(m.get("harga_angka",0)); usia=_as_int(m.get("usia",0))
        bb=str(m.get("bahan_bakar","")).lower()
        trans=str(m.get("transmisi","")).lower()
        merek=str(m.get("merek","")).lower()

        key=f"{nama.lower()}__{tahun}"
        if key in seen: continue
        seen.add(key)

        # filter metadata (longgar)
        if "bahan_bakar" in filters and filters["bahan_bakar"] not in bb: continue
        if "transmisi" in filters and filters["transmisi"]!=trans: continue
        if "merek" in filters and filters["merek"] not in merek: continue

        # numeric filter
        if "harga_max" in numeric and harga and harga>numeric["harga_max"]: continue
        if "tahun_min" in numeric and tahun and tahun<numeric["tahun_min"]: continue
        if "usia_max" in numeric and usia and usia>numeric["usia_max"]: continue

        hasil.append({
            "nama_mobil":nama,"tahun":tahun,"harga":m.get("harga","-"),
            "harga_angka":harga,"usia":usia,"bahan_bakar":bb,
            "transmisi":trans,"kapasitas_mesin":m.get("kapasitas_mesin","-"),
            "cosine_score":round(float(score),4)
        })

    # fallback: kalau kosong, ambil top-K tanpa filter
    if not hasil:
        hasil = [{
            "nama_mobil":doc.metadata.get("nama_mobil","-"),
            "tahun":_as_int(doc.metadata.get("tahun",0)),
            "harga":doc.metadata.get("harga","-"),
            "harga_angka":_as_int(doc.metadata.get("harga_angka",0)),
            "usia":_as_int(doc.metadata.get("usia",0)),
            "bahan_bakar":doc.metadata.get("bahan_bakar",""),
            "transmisi":doc.metadata.get("transmisi",""),
            "kapasitas_mesin":doc.metadata.get("kapasitas_mesin","-"),
            "cosine_score":round(float(score),4)
        } for doc,score in docs_scores[:k]]

    return {"rekomendasi":hasil[:k]}
