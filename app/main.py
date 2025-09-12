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
BUILD_EMBED_ON_START = os.getenv("BUILD_EMBED_ON_START", "0") == "1"
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

# --- Router rule-based ---
from app.rule_based import router as rule_router  # noqa: E402
from app.rule_based import jawab_rule  # untuk alias lama
app.include_router(rule_router)

# --- Opsional RAG/Chroma ---
if ENABLE_RAG:
    try:
        # Bangun embedding HANYA jika diizinkan & belum ada
        need_build = not CHROMA_PERSIST_DIR.exists() or not any(CHROMA_PERSIST_DIR.rglob("*"))
        if need_build and BUILD_EMBED_ON_START:
            print("[INIT] Chroma index belum ada -> generate embedding ...")
            from app.embedding import simpan_vektor_mobil  # noqa: E402
            simpan_vektor_mobil(persist_dir=str(CHROMA_PERSIST_DIR))
        else:
            print("[INIT] Chroma index:", "kosong" if need_build else "ditemukan", CHROMA_PERSIST_DIR)

        from app.rag_qa import router as rag_router  # noqa: E402
        app.include_router(rag_router)
    except Exception as e:
        print("[WARN] ENABLE_RAG=1 tapi gagal inisialisasi RAG:", e)

# Endpoint rebuild embedding (manual)
@app.post("/admin/rebuild-embeddings")
def rebuild(secret: str):
    if secret != os.getenv("REBUILD_SECRET", "dev"):
        return {"ok": False, "error": "unauthorized"}
    from app.embedding import simpan_vektor_mobil
    simpan_vektor_mobil(persist_dir=str(CHROMA_PERSIST_DIR))
    return {"ok": True, "persist_dir": str(CHROMA_PERSIST_DIR)}

# Alias untuk kompatibilitas UI lama
@app.get("/cosine_rekomendasi")
def cosine_alias(query: str):
    recs = jawab_rule(query, topk=10)
    mapped = []
    for r in recs:
        mapped.append({
            "nama_mobil": r["nama_mobil"],
            "tahun": r["tahun"],
            "harga": r["harga"],
            "usia": r["usia"],
            "bahan_bakar": r["bahan_bakar"],
            "transmisi": r["transmisi"],
            "kapasitas_mesin": r["kapasitas_mesin"],
            "cosine_score": r.get("skor") if r.get("skor") is not None else 0.0
        })
    return {"rekomendasi": mapped}
