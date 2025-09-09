import os
import shutil
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR   = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # longgarkan dulu agar mudah tes dari mana saja
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Serve frontend
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health(): return {"ok": True}

# ===== Admin: rebuild chroma (hapus folder lama, bangun ulang)
@app.post("/admin/rebuild_chroma")
def rebuild_chroma():
    try:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)  # bersih total
        from app.embedding import simpan_vektor_mobil
        simpan_vektor_mobil()
        return JSONResponse({"ok": True, "dir": str(CHROMA_DIR)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ===== Debug: lihat isi folder chroma
@app.get("/debug/chroma")
def debug_chroma():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p.name for p in CHROMA_DIR.glob("*")])
    return {"dir": str(CHROMA_DIR), "files": files}

# ===== (opsional) Statistik dataset
@app.get("/debug/dataset_stats")
def dataset_stats():
    df = pd.read_csv(DATA_CSV)
    cols = list(df.columns)
    fuels = df["Bahan Bakar"].astype(str).str.lower().value_counts().to_dict() if "Bahan Bakar" in df.columns else {}
    brands = df["Nama Mobil"].astype(str).str.split().str[0].str.lower().value_counts().head(15).to_dict() if "Nama Mobil" in df.columns else {}
    return {"columns": cols, "fuels": fuels, "top_brands": brands}
