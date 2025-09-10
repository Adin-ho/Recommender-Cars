import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# --- ENV ---
ENABLE_RAG = os.getenv("ENABLE_RAG", "1") == "1"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "chroma")))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

app = FastAPI(title="Car Recommender")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static (opsional)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Health
@app.get("/healthz", response_class=PlainTextResponse)
def health():
    return "ok"

# Home -> render frontend/index.html bila ada
@app.get("/")
def home():
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {"message": "Car Recommender API. Buka /docs untuk Swagger."}

# --- Import router setelah app ada ---
from app.rule_based import router as rule_router  # noqa: E402
app.include_router(rule_router)

if ENABLE_RAG:
    try:
        # Auto build embedding kalau persist directory belum ada atau kosong
        need_build = not CHROMA_PERSIST_DIR.exists() or not any(CHROMA_PERSIST_DIR.rglob("*"))
        if need_build:
            print("[INIT] Chroma index belum ada -> generate embedding ...")
            from app.embedding import simpan_vektor_mobil  # noqa: E402
            simpan_vektor_mobil(persist_dir=str(CHROMA_PERSIST_DIR))
        else:
            print("[INIT] Chroma index ditemukan:", CHROMA_PERSIST_DIR)
        from app.rag_qa import router as rag_router  # noqa: E402
        app.include_router(rag_router)
    except Exception as e:
        print("[WARN] ENABLE_RAG=1 tapi gagal inisialisasi RAG:", e)

# Endpoint rebuild embedding (opsional, lindungi dengan SECRET bila dipakai)
@app.post("/admin/rebuild-embeddings")
def rebuild(secret: str):
    if secret != os.getenv("REBUILD_SECRET", "dev"):
        return {"ok": False, "error": "unauthorized"}
    from app.embedding import simpan_vektor_mobil
    simpan_vektor_mobil(persist_dir=str(CHROMA_PERSIST_DIR))
    return {"ok": True, "persist_dir": str(CHROMA_PERSIST_DIR)}
