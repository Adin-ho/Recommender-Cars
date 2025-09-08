import os
import re
import asyncio
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ===== Path aman (berbasis file ini) =====
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))

if os.getenv("ENABLE_RAG", "0") == "1":
    try:
        from app.rag_qa import router as rag_qa_router
        app.include_router(rag_qa_router)

        if not CHROMA_DIR.exists():
            print("[INIT] chroma belum ada → generate embedding ke:", CHROMA_DIR)
            from app.embedding import simpan_vektor_mobil
            simpan_vektor_mobil()
        else:
            print("[INIT] chroma sudah ada di:", CHROMA_DIR)
    except Exception as e:
        print("[INIT] ENABLE_RAG=1 tapi gagal load RAG:", e)
app = FastAPI()

# ===== CORS bebas untuk demo =====
@app.get("/health")
def health():
    return {"ok": True}

# batasi asal CORS (ganti dengan domain kamu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://recommender-cars.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/admin/rebuild_chroma")
def rebuild_chroma():
    try:
        CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))
        # bersihkan folder index lama (aman karena akan diisi ulang)
        if CHROMA_DIR.exists():
            for p in CHROMA_DIR.glob("*"):
                if p.is_file(): p.unlink()
        from app.embedding import simpan_vektor_mobil
        simpan_vektor_mobil()
        return JSONResponse({"ok": True, "dir": str(CHROMA_DIR)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ===== Layani frontend =====
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    index_html = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_html))

# ===== Baca dataset =====
data_mobil = pd.read_csv(DATA_CSV)
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

if "harga_angka" not in data_mobil.columns:
    def bersihkan_harga(h):
        if pd.isna(h): return 0
        s = str(h)
        return int(re.sub(r"\D", "", s)) if re.search(r"\d", s) else 0
    data_mobil["harga_angka"] = data_mobil["harga"].apply(bersihkan_harga)

def unique_cars(output: str) -> str:
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

# ===== Endpoint streaming (SSE) =====
@app.get("/stream")
async def stream(pertanyaan: str, exclude: str = ""):
    jawaban_text = jawab(pertanyaan, exclude)  # panggil fungsi di bawah
    async def event_stream():
        for word in jawaban_text.split():
            yield f"data: {word}\n\n"
            await asyncio.sleep(0.06)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ===== Endpoint rule-based utama =====
@app.get("/jawab", response_class=PlainTextResponse)
def jawab(pertanyaan: str, exclude: str = ""):
    hasil = data_mobil.copy()
    q = pertanyaan.lower()
    tahun_sekarang = 2025

    # Usia
    m_usia = re.search(r"usia (?:di bawah|kurang dari) (\d+)\s*tahun", q)
    if m_usia:
        batas_tahun = tahun_sekarang - int(m_usia.group(1))
        hasil = hasil[hasil["tahun"] >= batas_tahun]

    # Transmisi
    if "matic" in q and "manual" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
    if "manual" in q and "matic" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]

    # Bahan bakar
    for bb in ["diesel", "bensin", "hybrid", "listrik"]:
        if bb in q:
            hasil = hasil[hasil["bahan bakar"].str.contains(bb, case=False, na=False)]

    # Harga (contoh: "di bawah 150.000.000" / "max 200000000")
    m_harga = re.search(r"(?:di bawah|max(?:imal)?|<=?) ?rp? ?(\d[\d\.]*)", q)
    if m_harga:
        batas = int(m_harga.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]

    # Tahun ke atas
    m_tahun_atas = re.search(r"tahun (\d{4}) ke atas", q)
    if m_tahun_atas:
        hasil = hasil[hasil["tahun"] >= int(m_tahun_atas.group(1))]

    # Tahun di bawah
    m_tahun_bawah = re.search(r"tahun (?:di bawah|kurang dari) (\d{4})", q)
    if m_tahun_bawah:
        hasil = hasil[hasil["tahun"] < int(m_tahun_bawah.group(1))]

    # Sinonim irit/hemat → bensin/hybrid
    if "irit" in q or "hemat" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("bensin|hybrid", case=False, na=False)]

    # Exclude list (nama mobil yang sudah ditampilkan)
    exclude_list = [x.strip().lower() for x in exclude.split(",") if x.strip()]
    if exclude_list:
        hasil = hasil[~hasil["nama mobil"].str.lower().isin(exclude_list)]

    if hasil.empty:
        return "tidak ditemukan"

    output = "; ".join(
        _bersih_nama(row["nama mobil"], row["tahun"])
        for _, row in hasil.head(5).iterrows()
    )
    return unique_cars(output)

# ===== (Opsional) RAG berbasis CPU =====
if os.getenv("ENABLE_RAG", "0") == "1":
    try:
        from app.rag_qa import router as rag_qa_router
        app.include_router(rag_qa_router)

        # Auto-bangun index Chroma kalau belum ada
        CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))
        if not CHROMA_DIR.exists():
            print("[INIT] chroma/ belum ada → generate embedding...")
            from app.embedding import simpan_vektor_mobil
            simpan_vektor_mobil()
        else:
            print("[INIT] chroma/ sudah ada.")
    except Exception as e:
        print("[INIT] ENABLE_RAG=1 tapi gagal load RAG:", e)
