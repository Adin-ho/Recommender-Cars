import os, re, random
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

_BRANDS = ["bmw","toyota","honda","suzuki","renault","wuling","mitsubishi","daihatsu","nissan",
           "mazda","hyundai","kia","vw","volkswagen","mercedes","audi"]

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

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(query:str=Query(...), k:int=5, exclude:str=""):
    docs_scores = VECTOR.similarity_search_with_score(query, k=80)
    filters=_parse_filters(query)
    numeric=_parse_numeric(query)

    hasil=[]
    seen=set()
    for doc,score in docs_scores:
        m=doc.metadata or {}
        nama=m.get("nama_mobil","-"); tahun=m.get("tahun",0)
        harga=int(m.get("harga_angka",0)); usia=int(m.get("usia",0))
        bb=m.get("bahan_bakar",""); trans=m.get("transmisi",""); merek=m.get("merek","")

        key=f"{nama.lower()}__{tahun}"
        if key in seen: continue
        seen.add(key)

        # filter metadata
        if filters.get("bahan_bakar") and filters["bahan_bakar"] not in bb: continue
        if filters.get("transmisi") and filters["transmisi"]!=trans: continue
        if filters.get("merek") and filters["merek"] not in merek: continue

        # numeric
        if "harga_max" in numeric and harga and harga>numeric["harga_max"]: continue
        if "tahun_min" in numeric and tahun and tahun<numeric["tahun_min"]: continue
        if "usia_max" in numeric and usia and usia>numeric["usia_max"]: continue

        hasil.append({
            "nama_mobil":nama,"tahun":tahun,"harga":m.get("harga","-"),
            "harga_angka":harga,"usia":usia,"bahan_bakar":bb,
            "transmisi":trans,"kapasitas_mesin":m.get("kapasitas_mesin","-"),
            "cosine_score":round(float(score),4)
        })
    if not hasil: return {"jawaban":"Maaf, tidak ditemukan.","rekomendasi":[]}
    return {"rekomendasi":hasil[:k]}
