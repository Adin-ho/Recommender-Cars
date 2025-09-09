import os
import re
import asyncio
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------- PATH ----------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cars")

app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sesuaikan jika mau dikunci
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- STATIC ----------
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health():
    return {"ok": True}

# ---------- DATASET ----------
data_mobil = pd.read_csv(DATA_CSV)
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

if "harga_angka" not in data_mobil.columns:
    def _clean_harga(h):
        if pd.isna(h): return 0
        s = str(h)
        return int(re.sub(r"\D", "", s)) if re.search(r"\d", s) else 0
    data_mobil["harga_angka"] = data_mobil["harga"].apply(_clean_harga)

# ---------- RULE-BASED ----------
def _bersih_nama(nama: str, tahun: int) -> str:
    nama = re.sub(r"\s*\(\d{4}\)$", "", str(nama).strip().lower())
    return f"{nama} ({tahun})"

def unique_cars(output: str) -> str:
    found = re.findall(r"([a-z0-9 .\-]+)\s*\((\d{4})\)", output.lower())
    seen, cars = set(), []
    for n, t in found:
        key = f"{n.strip()} ({t})"
        if key not in seen:
            seen.add(key)
            cars.append(key)
    return "; ".join(cars)

@app.get("/jawab", response_class=PlainTextResponse)
def jawab(pertanyaan: str, exclude: str = ""):
    hasil = data_mobil.copy()
    q = pertanyaan.lower()
    tahun_sekarang = 2025

    # usia
    m_usia = re.search(r"usia (?:di bawah|kurang dari) (\d+)\s*tahun", q)
    if m_usia:
        batas_tahun = tahun_sekarang - int(m_usia.group(1))
        hasil = hasil[hasil["tahun"] >= batas_tahun]

    # transmisi
    if "matic" in q and "manual" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
    if "manual" in q and "matic" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]

    # bahan bakar
    for bb in ["diesel", "bensin", "hybrid", "listrik"]:
        if bb in q and "bahan bakar" in data_mobil.columns:
            hasil = hasil[hasil["bahan bakar"].str.contains(bb, case=False, na=False)]

    # harga (contoh: "di bawah 150.000.000", "max 200000000")
    m_harga = re.search(r"(?:di bawah|max(?:imal)?|<=?) ?rp? ?(\d[\d\.]*)", q)
    if m_harga:
        batas = int(m_harga.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]

    # tahun ke atas
    m_tahun_atas = re.search(r"tahun (\d{4}) ke atas", q)
    if m_tahun_atas:
        hasil = hasil[hasil["tahun"] >= int(m_tahun_atas.group(1))]

    # tahun di bawah
    m_tahun_bawah = re.search(r"tahun (?:di bawah|kurang dari) (\d{4})", q)
    if m_tahun_bawah:
        hasil = hasil[hasil["tahun"] < int(m_tahun_bawah.group(1))]

    # sinonim irit/hemat -> bensin/hybrid
    if "irit" in q or "hemat" in q:
        if "bahan bakar" in data_mobil.columns:
            hasil = hasil[hasil["bahan bakar"].str.contains("bensin|hybrid", case=False, na=False)]

    # exclude list
    exclude_list = [x.strip().lower() for x in exclude.split(",") if x.strip()]
    if exclude_list and "nama mobil" in data_mobil.columns:
        hasil = hasil[~hasil["nama mobil"].str.lower().isin(exclude_list)]

    if hasil.empty:
        return "tidak ditemukan"

    output = "; ".join(
        _bersih_nama(row["nama mobil"], row["tahun"])
        for _, row in hasil.head(5).iterrows()
    )
    return unique_cars(output)

# ---------- RAG (Chroma) ----------
USE_RAG = os.getenv("ENABLE_RAG", "0") == "1"
if USE_RAG:
    from app.rag_qa import router as rag_router
    app.include_router(rag_router)

    # cek isi koleksi secara nyata (bukan hanya cek folder)
    def _chroma_doc_count() -> int:
        try:
            from langchain_chroma import Chroma
            vs = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR))
            # `count()` tersedia di client baru; fallback ke len(get()) untuk amannya
            try:
                return vs._collection.count()  # type: ignore
            except Exception:
                return len(vs.get()["ids"])
        except Exception:
            return 0

    # bangun embedding kalau kosong
    try:
        count = _chroma_doc_count()
        if count <= 0:
            print(f"[INIT] Koleksi kosong -> build embedding ke: {CHROMA_DIR}")
            from app.embedding import simpan_vektor_mobil
            simpan_vektor_mobil(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
        else:
            print(f"[INIT] Chroma OK. Dokumen: {count} di {CHROMA_DIR}")
    except Exception as e:
        print("[INIT] Gagal inisialisasi Chroma:", e)

    # endpoint debug
    @app.get("/debug/chroma_count")
    def chroma_count():
        try:
            return {"dir": str(CHROMA_DIR), "collection": COLLECTION_NAME, "count": _chroma_doc_count()}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # admin: rebuild manual
    @app.post("/admin/rebuild_chroma")
    def rebuild():
        try:
            from app.embedding import simpan_vektor_mobil
            # bersihkan isi lama
            if CHROMA_DIR.exists():
                for p in CHROMA_DIR.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
            simpan_vektor_mobil(collection_name=COLLECTION_NAME, persist_dir=str(CHROMA_DIR))
            return {"ok": True, "dir": str(CHROMA_DIR), "count": _chroma_doc_count()}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---------- SSE contoh (opsional) ----------
@app.get("/stream")
async def stream(pertanyaan: str, exclude: str = ""):
    text = jawab(pertanyaan, exclude)
    async def gen():
        for w in str(text).split():
            yield f"data: {w}\n\n"
            await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="text/event-stream")
