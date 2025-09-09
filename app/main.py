import os, shutil, re, asyncio
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR   = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))

app = FastAPI()

# CORS longgar biar gampang tes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== ROUTER RAG (selalu di-mount)
from app.rag_qa import router as rag_router
app.include_router(rag_router)

# Auto-bangun index kalau kosong
if not CHROMA_DIR.exists() or not any(CHROMA_DIR.glob("*")):
    print("[INIT] Chroma kosong → generate embedding…")
    from app.embedding import simpan_vektor_mobil
    simpan_vektor_mobil()
else:
    print(f"[INIT] Chroma sudah ada di: {CHROMA_DIR}")

# ==== Frontend
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health():
    return {"ok": True}

# ==== Admin: rebuild index bersih
@app.post("/admin/rebuild_chroma")
def rebuild_chroma():
    try:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        from app.embedding import simpan_vektor_mobil
        simpan_vektor_mobil()
        return JSONResponse({"ok": True, "dir": str(CHROMA_DIR)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ==== Endpoint debug kecil
@app.get("/debug/chroma")
def debug_chroma():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return {"dir": str(CHROMA_DIR), "files": sorted(p.name for p in CHROMA_DIR.glob("*"))}

@app.get("/debug/chroma_count")
def debug_chroma_count():
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    encode_kwargs={"normalize_embeddings": True})
        vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)
        return {"dir": str(CHROMA_DIR), "count": vs._collection.count()}
    except Exception as e:
        return {"dir": str(CHROMA_DIR), "count": None, "error": str(e)}

@app.get("/debug/dataset_stats")
def dataset_stats():
    df = pd.read_csv(DATA_CSV)
    fuels = df["Bahan Bakar"].astype(str).str.lower().value_counts().to_dict() if "Bahan Bakar" in df.columns else {}
    brands = df["Nama Mobil"].astype(str).str.split().str[0].str.lower().value_counts().head(15).to_dict() if "Nama Mobil" in df.columns else {}
    return {"rows": len(df), "fuels": fuels, "top_brands": brands}

# ==== (opsional) Rule-based bawaan untuk endpoint /jawab
data_mobil = pd.read_csv(DATA_CSV)
data_mobil.columns = data_mobil.columns.str.strip().str.lower()
if "harga_angka" not in data_mobil.columns:
    def _harga(h):
        if pd.isna(h): return 0
        s = str(h)
        m = re.sub(r"\D", "", s)
        return int(m) if m else 0
    data_mobil["harga_angka"] = data_mobil["harga"].apply(_harga)

def _bersih_nama(nama: str, tahun: int) -> str:
    nama = re.sub(r"\s*\(\d{4}\)$", "", str(nama).strip().lower())
    return f"{nama} ({tahun})"

def _unique(output: str) -> str:
    found = re.findall(r"([a-z0-9 .\-]+)\s*\((\d{4})\)", output.lower())
    seen, cars = set(), []
    for n,t in found:
        key = f"{n.strip()} ({t})"
        if key not in seen:
            seen.add(key); cars.append(key)
    return "; ".join(cars)

@app.get("/jawab", response_class=PlainTextResponse)
def jawab(pertanyaan: str, exclude: str = ""):
    q = pertanyaan.lower()
    hasil = data_mobil.copy()
    thn_now = 2025

    if m := re.search(r"usia (?:di bawah|kurang dari) (\d+)\s*tahun", q):
        hasil = hasil[hasil["tahun"] >= thn_now - int(m.group(1))]
    if "matic" in q and "manual" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
    if "manual" in q and "matic" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]
    for bb in ["diesel","bensin","hybrid","listrik"]:
        if bb in q:
            hasil = hasil[hasil["bahan bakar"].str.contains(bb, case=False, na=False)]
    if m := re.search(r"(?:di bawah|max(?:imal)?|<=?) ?rp? ?(\d[\d\.]*)", q):
        hasil = hasil[hasil["harga_angka"] <= int(m.group(1).replace(".",""))]
    if m := re.search(r"tahun (\d{4}) ke atas", q):
        hasil = hasil[hasil["tahun"] >= int(m.group(1))]
    if m := re.search(r"tahun (?:di bawah|kurang dari) (\d{4})", q):
        hasil = hasil[hasil["tahun"] < int(m.group(1))]
    if "irit" in q or "hemat" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("bensin|hybrid", case=False, na=False)]

    ex = [x.strip().lower() for x in exclude.split(",") if x.strip()]
    if ex:
        hasil = hasil[~hasil["nama mobil"].str.lower().isin(ex)]

    if hasil.empty:
        return "tidak ditemukan"
    out = "; ".join(_bersih_nama(r["nama mobil"], r["tahun"]) for _,r in hasil.head(5).iterrows())
    return _unique(out)

# ==== SSE demo (opsional)
@app.get("/stream")
async def stream(pertanyaan: str, exclude: str = ""):
    text = jawab(pertanyaan, exclude)
    async def _gen():
        for w in text.split():
            yield f"data: {w}\n\n"
            await asyncio.sleep(0.06)
    return StreamingResponse(_gen(), media_type="text/event-stream")
