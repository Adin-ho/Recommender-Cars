# app/main.py
from pathlib import Path
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR = ROOT_DIR / "chroma"   # di-ignore dari git

app = FastAPI(title="ChatCars API")

# CORS (aman utk demo / Space)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend statis di root "/"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

RAG_READY = False

@app.on_event("startup")
async def build_index_if_needed():
    """Bangun index Chroma otomatis saat start (sekali saja)."""
    global RAG_READY
    try:
        from .embedding import ensure_chroma
        ensure_chroma(csv_path=DATA_CSV, persist_dir=CHROMA_DIR)
        RAG_READY = True
        print("[INIT] RAG index ready")
    except Exception as e:
        RAG_READY = False
        print(f"[INIT] ENABLE_RAG=1 tapi gagal init RAG: {e}")

@app.get("/healthz")
def healthz():
    return {"ok": True, "rag_ready": RAG_READY}

# ---- Endpoint rekomendasi ----

@app.get("/cosine_rekomendasi")
def api_cosine_rekomendasi(query: str = Query(..., min_length=1), top_k: int = 5):
    """
    Coba pakai RAG (Chroma). Jika tidak siap / error → fallback ke rule-based.
    Response diseragamkan: {source: "rag"|"rule_based", items: [...]}
    """
    # 1) Coba RAG
    if RAG_READY:
        try:
            from .rag_qa import cosine_rekomendasi_rag
            items = cosine_rekomendasi_rag(query, top_k=top_k, csv_path=DATA_CSV, persist_dir=CHROMA_DIR)
            return {"source": "rag", "items": items}
        except Exception as e:
            # catat & teruskan ke fallback
            print(f"[RAG] error: {e}")

    # 2) Fallback rule-based (SELALU ada jawaban)
    try:
        from .rule_based import rekomendasi_rule_based
        items = rekomendasi_rule_based(query, csv_path=DATA_CSV, top_k=top_k)
        return {"source": "rule_based", "items": items}
    except Exception as e:
        # supaya frontend tidak hanya "gagal mengambil jawaban" tanpa info
        return JSONResponse({"error": f"Gagal memproses query: {e}"}, status_code=400)
