import os
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

ENABLE_LLM = os.getenv("ENABLE_LLM", "1") == "1"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

app = FastAPI(title="Recommender Cars (Zeabur + Ollama Mistral)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⬇️ Mount frontend hanya di /static agar /api TIDAK ketimpa
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Health
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# Home: render index.html dari frontend/
@app.get("/")
def home():
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {"message": "Car Recommender API. Open /docs for Swagger."}

# === Rule-based API (selalu ON)
from app.rule_based import router as rule_router, jawab_rule  # noqa: E402
app.include_router(rule_router)

# === LLM Router (opsional, expose /api/llm/chat)
if ENABLE_LLM:
    try:
        from app.llm_proxy import router as llm_router, ollama_chat  # noqa: E402
        app.include_router(llm_router)
        print("[INIT] LLM router aktif (Ollama Mistral)")
    except Exception as e:
        print("[WARN] ENABLE_LLM=1 tapi gagal load llm_proxy:", e)
        ollama_chat = None
else:
    ollama_chat = None

# === Endpoint gabungan: rule-based + (opsional) ringkasan Mistral
@app.get("/api/ask")
async def api_ask(
    pertanyaan: str = Query(..., description="Contoh: 'mobil listrik matic di bawah 500 jt'"),
    topk: int = Query(5, ge=1, le=50)
):
    recs = jawab_rule(pertanyaan, topk=topk)

    if not recs:
        return {
            "jawaban": "Tidak ditemukan.",
            "rekomendasi": [],
            "llm_model": os.getenv("OLLAMA_MODEL", "mistral") if ENABLE_LLM else None
        }

    plain = "Hasil rekomendasi:\n\n" + "\n".join([
        f"{i+1}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - "
        f"{r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}"
        for i, r in enumerate(recs)
    ])

    # Ringkas pakai Mistral (jika diaktifkan dan koneksi ada)
    if ENABLE_LLM and callable(ollama_chat):
        items = "\n".join([
            f"{i+1}. {r['nama_mobil']} ({r['tahun']}), harga {r['harga']}, "
            f"bahan bakar {r['bahan_bakar']}, transmisi {r['transmisi']}, "
            f"kapasitas {r['kapasitas_mesin']}, usia {r['usia']} tahun"
            for i, r in enumerate(recs)
        ])
        prompt = (
            "Anda adalah asisten showroom mobil bekas. "
            "Ringkas rekomendasi berdasarkan pertanyaan dan daftar hasil berikut. "
            "Jangan mengarang data baru. Maks 5 bullet.\n\n"
            f"Pertanyaan: {pertanyaan}\n\nDaftar:\n{items}\n\n"
            "Sorot kecocokan (bahan bakar, transmisi, harga) dan beri saran singkat."
        )
        try:
            llm_text = await ollama_chat(prompt)
        except Exception as e:
            llm_text = f"(LLM tidak tersedia: {e})"
    else:
        llm_text = None

    return JSONResponse({
        "jawaban": llm_text or plain,
        "rekomendasi": recs,
        "llm_model": os.getenv("OLLAMA_MODEL", "mistral") if llm_text else None
    })
