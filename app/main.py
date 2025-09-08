# app/main.py
from pathlib import Path
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

# >>> PENTING: simpan index ke folder yang writable di HF
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/data/chroma"))

FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="ChatCars API")

# CORS longgar untuk demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_READY = False

@app.on_event("startup")
async def _startup():
    """Bangun index Chroma otomatis (sekali saja)."""
    global RAG_READY
    try:
        from .embedding import ensure_chroma
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        ensure_chroma(csv_path=DATA_CSV, persist_dir=CHROMA_DIR)
        RAG_READY = True
        print(f"[INIT] RAG index ready at {CHROMA_DIR}")
    except Exception as e:
        RAG_READY = False
        print(f"[INIT] RAG disabled: {e}")

@app.get("/healthz")
def healthz():
    return {"ok": True, "rag_ready": RAG_READY, "chroma_dir": str(CHROMA_DIR)}

# --------- Endpoint utama ----------
@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(query: str = Query(..., min_length=1), top_k: int = 5):
    """
    Coba pakai RAG; kalau gagal → fallback ke rule-based
    Output diseragamkan: {source: "...", items: [...]}
    """
    # 1) RAG
    if RAG_READY:
        try:
            from .rag_qa import cosine_rekomendasi_rag
            items = cosine_rekomendasi_rag(
                query=query, top_k=top_k, csv_path=DATA_CSV, persist_dir=CHROMA_DIR
            )
            return {"source": "rag", "items": items}
        except Exception as e:
            print(f"[RAG] error: {e}")  # lanjut ke fallback

    # 2) Fallback rule-based (selalu ada jawaban)
    try:
        from .rule_based import rekomendasi_rule_based
        items = rekomendasi_rule_based(query=query, csv_path=DATA_CSV, top_k=top_k)
        return {"source": "rule_based", "items": items}
    except Exception as e:
        return JSONResponse({"error": f"Gagal memproses query: {e}"}, status_code=400)

# ---- Terakhir: mount frontend (supaya API tidak ketutup) ----
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
