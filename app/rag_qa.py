import os, re, random
from pathlib import Path
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", Path(__file__).resolve().parents[1] / "chroma"))

EMB = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
VS = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=EMB)
router = APIRouter()

_BRANDS = ["bmw","toyota","honda","suzuki","daihatsu","nissan","mitsubishi",
           "mazda","hyundai","kia","wuling","renault","vw","volkswagen","mercedes","audi"]

def _i(x, d=0):
    try: return int(float(x))
    except: return d

def _parse_query(q:str):
    ql = q.lower()
    harga_target = None
    if m := re.search(r"(\d{2,4})\s*juta", ql): harga_target = int(m.group(1))*1_000_000
    elif m := re.search(r"(\d{9,12})", ql.replace(".","")): harga_target = int(m.group(1))
    usia_max = 5
    if m := re.search(r"(?:<|di ?bawah|kurang dari|max(?:imal)?)\s*(\d{1,2})\s*tahun", ql):
        usia_max = int(m.group(1))
    fuel = None
    for bb in ["listrik","electric","diesel","bensin","hybrid"]:
        if bb in ql: fuel = "listrik" if bb=="electric" else bb; break
    trans = "matic" if "matic" in ql and "manual" not in ql else ("manual" if "manual" in ql and "matic" not in ql else None)
    brand = None
    for b in _BRANDS:
        if b in ql: brand = "volkswagen" if b=="vw" else b; break
    return harga_target, usia_max, fuel, trans, brand

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(
    query: str = Query(..., description="Contoh: 'rekomendasi mobil diesel 300 juta'"),
    k: int = Query(5, ge=1, le=50),
    exclude: str = ""
):
    harga_target, usia_max, fuel, trans, brand = _parse_query(query)
    raw = VS.similarity_search_with_score(query, k=max(80, 8*k))

    ex = {x.strip().lower() for x in exclude.split(",") if x.strip()}
    seen, utama, cad1, cad2 = set(), [], [], []
    tol = 0.18
    hmin = int(harga_target*(1-tol)) if harga_target else 0
    hmax = int(harga_target*(1+tol)) if harga_target else 10**12

    for doc, score in raw:
        m = doc.metadata or {}
        nama = str(m.get("nama_mobil","-"))
        key  = f"{nama.lower().strip()}__{m.get('tahun','-')}"
        if nama.lower() in ex or key in seen: continue
        seen.add(key)

        row = {
            "nama_mobil": nama,
            "tahun": _i(m.get("tahun",0)),
            "harga": m.get("harga","-"),
            "harga_angka": _i(m.get("harga_angka",0)),
            "usia": _i(m.get("usia",0)),
            "bahan_bakar": str(m.get("bahan_bakar","")).lower(),
            "transmisi": str(m.get("transmisi","")),
            "kapasitas_mesin": m.get("kapasitas_mesin","-"),
            "merek": str(m.get("merek","")),
            "cosine_score": round(float(score), 4),
        }

        # filter longgar (hanya kalau metadata ada)
        if fuel and row["bahan_bakar"] and fuel not in row["bahan_bakar"]: continue
        if trans and row["transmisi"] and trans != row["transmisi"]: continue
        if brand and row["merek"] and brand not in row["merek"]: continue

        if harga_target:
            in_range = row["harga_angka"] and hmin <= row["harga_angka"] <= hmax
            if in_range and (0 < row["usia"] <= usia_max):   utama.append(row)
            elif in_range and row["usia"] > usia_max:         cad1.append(row)
            else:                                             cad2.append(row)
        else:
            if 0 < row["usia"] <= usia_max: utama.append(row)
            else:                           cad2.append(row)

    utama.sort(key=lambda x: (abs((harga_target or 0)-x["harga_angka"]), x["usia"]))
    cad1.sort(key=lambda x: (x["usia"], abs((harga_target or 0)-x["harga_angka"])))
    random.shuffle(cad2)

    hasil = (utama + cad1 + cad2)[:k]
    if not hasil and raw:   # Fallback: jangan pernah kosong
        hasil = [{
            "nama_mobil": (d.metadata or {}).get("nama_mobil","-"),
            "tahun": _i((d.metadata or {}).get("tahun",0)),
            "harga": (d.metadata or {}).get("harga","-"),
            "harga_angka": _i((d.metadata or {}).get("harga_angka",0)),
            "usia": _i((d.metadata or {}).get("usia",0)),
            "bahan_bakar": str((d.metadata or {}).get("bahan_bakar","")).lower(),
            "transmisi": str((d.metadata or {}).get("transmisi","")),
            "kapasitas_mesin": (d.metadata or {}).get("kapasitas_mesin","-"),
            "merek": str((d.metadata or {}).get("merek","")),
            "cosine_score": round(float(s),4),
        } for d,s in raw[:k]]

    return {"rekomendasi": hasil}

@router.get("/debug/chroma_count")
def chroma_count():
    try: return {"count": VS._collection.count(), "dir": str(PERSIST_DIR)}
    except Exception as e: return {"count": None, "dir": str(PERSIST_DIR), "error": str(e)}
