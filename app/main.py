import os, shutil
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR   = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

from app.rag_qa import router as rag_router
app.include_router(rag_router)

if not CHROMA_DIR.exists() or not any(CHROMA_DIR.glob("*")):
    print("[INIT] Chroma kosong → generate embedding…")
    from app.embedding import simpan_vektor_mobil
    simpan_vektor_mobil()
else:
    print(f"[INIT] Chroma sudah ada di: {CHROMA_DIR}")

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root(): return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health(): return {"ok": True}

@app.post("/admin/rebuild_chroma")
def rebuild_chroma():
    try:
        if CHROMA_DIR.exists(): shutil.rmtree(CHROMA_DIR)
        from app.embedding import simpan_vektor_mobil
        simpan_vektor_mobil()
        return JSONResponse({"ok": True, "dir": str(CHROMA_DIR)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/debug/chroma")
def debug_chroma():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return {"dir": str(CHROMA_DIR), "files": sorted(p.name for p in CHROMA_DIR.glob("*"))}

@app.get("/debug/dataset_stats")
def dataset_stats():
    csv = APP_DIR / "data" / "data_mobil_final.csv"
    df = pd.read_csv(csv)
    fuels = df["Bahan Bakar"].astype(str).str.lower().value_counts().to_dict() if "Bahan Bakar" in df.columns else {}
    brands= df["Nama Mobil"].astype(str).str.split().str[0].str.lower().value_counts().head(15).to_dict() if "Nama Mobil" in df.columns else {}
    return {"rows": len(df), "fuels": fuels, "top_brands": brands}
