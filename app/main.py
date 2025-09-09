# app/main.py
import os
import re
import asyncio
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    PlainTextResponse,
    FileResponse,
    StreamingResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles

# ===================== Paths & Config =====================
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

# Chroma path: pakai ENV kalau ada, fallback ke ./chroma
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma")).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cars")

# Domain asal front-end (kalau domainmu beda, tambah di sini—atau biarkan '*' saja)
ALLOWED_ORIGINS: List[str] = ["*"]

# ===================== FastAPI App =====================
app = FastAPI(title="ChatCars")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Layani folder frontend dan jadikan "/" = index.html
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health():
    return {"ok": True}

# ===================== Dataset & Rule-based =====================
if not DATA_CSV.exists():
    raise FileNotFoundError(f"CSV data tidak ditemukan: {DATA_CSV}")

data_mobil = pd.read_csv(DATA_CSV)
# normalisasi nama kolom
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

# kolom bantu: harga_angka
if "harga_angka" not in data_mobil.columns:
    def _bersihkan_harga(h):
        if pd.isna(h):
            return 0
        s = str(h)
        digits = re.sub(r"\D", "", s)
        return int(digits) if digits else 0
    if "harga" in data_mobil.columns:
        data_mobil["harga_angka"] = data_mobil["harga"].apply(_bersihkan_harga)
    else:
        data_mobil["harga_angka"] = 0

def _unique_cars(output: str) -> str:
    found = re.findall(r"([a-z0-9 .\-]+)\s*\((\d{4})\)", output.lower())
    seen, cars = set(), []
    for n, t in found:
        key = f"{n.strip()} ({t})"
        if key not in seen:
            seen.add(key)
            cars.append(key)
    return "; ".join(cars)

def _bersih_nama(nama: str, tahun: int) -> str:
    nama = re.sub(r"\s*\(\d{4}\)$", "", str(nama).strip().lower())
    return f"{nama} ({tahun})"

@app.get("/jawab", response_class=PlainTextResponse)
def jawab(pertanyaan: str, exclude: str = ""):
    hasil = data_mobil.copy()
    q = pertanyaan.lower()
    tahun_sekarang = 2025

    # usia
    m_usia = re.search(r"usia (?:di bawah|kurang dari) (\d+)\s*tahun", q)
    if m_usia and "tahun" in hasil.columns:
        batas_tahun = tahun_sekarang - int(m_usia.group(1))
        hasil = hasil[hasil["tahun"] >= batas_tahun]

    # transmisi
    if "transmisi" in hasil.columns:
        if "matic" in q and "manual" not in q:
            hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
        if "manual" in q and "matic" not in q:
            hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]

    # bahan bakar
    if "bahan bakar" in hasil.columns:
        for bb in ["diesel", "bensin", "hybrid", "listrik"]:
            if bb in q:
                hasil = hasil[hasil["bahan bakar"].str.contains(bb, case=False, na=False)]

    # harga (contoh: "di bawah 150.000.000", "max 200000000")
    m_harga = re.search(r"(?:di bawah|max(?:imal)?|<=?) ?rp? ?(\d[\d\.]*)", q)
    if m_harga and "harga_angka" in hasil.columns:
        batas = int(m_harga.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]

    # tahun ke atas
    m_tahun_atas = re.search(r"tahun (\d{4}) ke atas", q)
    if m_tahun_atas and "tahun" in hasil.columns:
        hasil = hasil[hasil["tahun"] >= int(m_tahun_atas.group(1))]

    # tahun di bawah
    m_tahun_bawah = re.search(r"tahun (?:di bawah|kurang dari) (\d{4})", q)
    if m_tahun_bawah and "tahun" in hasil.columns:
        hasil = hasil[hasil["tahun"] < int(m_tahun_bawah.group(1))]

    # sinonim irit/hemat -> bensin/hybrid
    if ("irit" in q or "hemat" in q) and "bahan bakar" in hasil.columns:
        hasil = hasil[hasil["bahan bakar"].str.contains("bensin|hybrid", case=False, na=False)]

    # exclude list
    if "nama mobil" in hasil.columns:
        exclude_list = [x.strip().lower() for x in exclude.split(",") if x.strip()]
        if exclude_list:
            hasil = hasil[~hasil["nama mobil"].str.lower().isin(exclude_list)]

    if hasil.empty:
        return "tidak ditemukan"

    output = "; ".join(
        _bersih_nama(row.get("nama mobil", ""), int(row.get("tahun", 0)))
        for _, row in hasil.head(5).iterrows()
    )
    return _unique_cars(output)

# Streaming contoh (SSE)
@app.get("/stream")
async def stream(pertanyaan: str, exclude: str = ""):
    jawaban_text = jawab(pertanyaan, exclude)
    async def event_stream():
        for word in jawaban_text.split():
            yield f"data: {word}\n\n"
            await asyncio.sleep(0.06)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ===================== RAG + Chroma =====================
USE_RAG = os.getenv("ENABLE_RAG", "0") == "1"
if USE_RAG:
    # router untuk endpoint cosine_rekomendasi
    from app.rag_qa import router as rag_router
    app.include_router(rag_router)

    # Cek jumlah dokumen di koleksi (bukan sekadar cek folder kosong)
    def _chroma_doc_count() -> int:
        try:
            from langchain_chroma import Chroma
            vs = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR))
            try:
                return vs._collection.count()  # chroma>=0.5
            except Exception:
                return len(vs.get()["ids"])
        except Exception:
            return 0

    # Auto build embedding kalau kosong
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        count = _chroma_doc_count()
        if count <= 0:
            print(f"[INIT] Koleksi kosong → build embedding ke: {CHROMA_DIR}")
            from app.embedding import simpan_vektor_mobil
            simpan_vektor_mobil(
                collection_name=COLLECTION_NAME,
                persist_dir=str(CHROMA_DIR)
            )
        else:
            print(f"[INIT] Chroma OK. Dokumen: {count} di {CHROMA_DIR}")
    except Exception as e:
        print("[INIT] Gagal inisialisasi Chroma:", e)

    # endpoint debug
    @app.get("/debug/chroma_count")
    def chroma_count():
        try:
            return {
                "dir": str(CHROMA_DIR),
                "collection": COLLECTION_NAME,
                "ok": True,
                "count": _chroma_doc_count(),
            }
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ===================== Admin & Debug Helpers =====================
@app.post("/admin/rebuild_chroma")
def rebuild_chroma():
    """Bangun ulang index vektor dari CSV."""
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        # bersihkan isi lama (file-file di folder)
        for p in CHROMA_DIR.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        from app.embedding import simpan_vektor_mobil
        simpan_vektor_mobil(
            collection_name=COLLECTION_NAME,
            persist_dir=str(CHROMA_DIR)
        )
        return JSONResponse({"ok": True, "dir": str(CHROMA_DIR)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/debug/chroma")
def debug_chroma():
    files = [p.name for p in CHROMA_DIR.glob("*")]
    return {"dir": str(CHROMA_DIR), "files": files}
