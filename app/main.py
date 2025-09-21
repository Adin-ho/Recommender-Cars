from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .rag_qa import cosine_rekomendasi
from .rule_based import router as rule_router
from .llm_proxy import router as llm_router

HERE = Path(__file__).resolve().parent
FRONT_DIR = HERE.parent / "frontend"
INDEX_HTML = FRONT_DIR / "index.html"

app = FastAPI(title="ChatCars")

# ===== CORS =====
ALLOWED = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in ALLOWED.split(",")] if ALLOWED else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Routers =====
app.include_router(rule_router)
app.include_router(llm_router)

# ===== Static Frontend =====
# (opsional mount folder; index dilayani manual supaya 200 OK di "/")
if FRONT_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONT_DIR)), name="static")

@app.get("/")
def home():
    return FileResponse(str(INDEX_HTML))

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ===== Cosine endpoint =====
@app.get("/cosine_rekomendasi")
def api_cosine_rekomendasi(query: str = Query(...), k: int = Query(5, ge=1, le=50)):
    try:
        result = cosine_rekomendasi(query, k)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
