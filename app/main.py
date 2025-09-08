# app/main.py
from pathlib import Path
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

# simpan index ke folder writable HF
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/data/chroma"))
FRONTEND_DIR = ROOT_DIR / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

app = FastAPI(title="ChatCars API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_READY = False

@app.on_event("startup")
async def _startup():
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

@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(query: str, top_k: int = 5):
    if not query:
        raise HTTPException(status_code=400, detail="query kosong")

    # 1) coba RAG
    if RAG_READY:
        try:
            from .rag_qa import cosine_rekomendasi_rag
            items = cosine_rekomendasi_rag(query=query, top_k=top_k, csv_path=DATA_CSV, persist_dir=CHROMA_DIR)
            return {"source": "rag", "rekomendasi": items}
        except Exception as e:
            print(f"[RAG] error: {e}")

    # 2) fallback rule-based
    try:
        from .rule_based import rekomendasi_rule_based
        items = rekomendasi_rule_based(query=query, csv_path=DATA_CSV, top_k=top_k)
        return {"source": "rule_based", "rekomendasi": items}
    except Exception as e:
        return JSONResponse({"error": f"Gagal memproses query: {e}"}, status_code=400)

# ===== UI =====
# Root PASTI mengembalikan index.html (fallback eksplisit)
@app.get("/", include_in_schema=False)
def root_page():
    if not INDEX_HTML.exists():
        # biar kelihatan jelas kalau file belum ter-copy
        return JSONResponse({"detail": "index.html tidak ditemukan di /frontend"}, status_code=500)
    return FileResponse(INDEX_HTML)

# Static di /static (opsional kalau kamu punya asset lain)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
