import os
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

# === Path dasar ===
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# === ENV flags ===
ENABLE_LLM = os.getenv("ENABLE_LLM", "1") == "1"   # default ON (boleh matikan)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# === Buat APP ===
app = FastAPI(title="Recommender Cars (Zeabur + Ollama Mistral)")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Static/frontend ===
if FRONTEND_DIR.exists():
    # Mount di root agar GET / merender index.html
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# === Health ===
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# === Rule-based API (selalu ON) ===
from app.rule_based import router as rule_router, jawab_rule  # noqa: E402
app.include_router(rule_router)

# === LLM Router (opsional, expose /api/llm/chat) ===
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

# === Endpoint gabungan: rule-based + (opsional) Mistral untuk merangkum ===
@app.get("/api/ask")
async def api_ask(
    pertanyaan: str = Query(..., description="Contoh: 'mobil listrik matic di bawah 500 jt'"),
    topk: int = Query(5, ge=1, le=50)
):
    recs = jawab_rule(pertanyaan, topk=topk)

    # Jika tidak ada hasil, langsung balikan
    if not recs:
        return {
            "jawaban": "Tidak ditemukan.",
            "rekomendasi": [],
            "llm_model": os.getenv("OLLAMA_MODEL", "mistral") if ENABLE_LLM else None
        }

    # Teks sederhana default (kalau LLM mati)
    plain = "Hasil rekomendasi:\n\n" + "\n".join([
        f"{i+1}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - "
        f"{r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}"
        for i, r in enumerate(recs)
    ])

    # Jika LLM aktif & tersedia, minta Mistral merangkum/menjelaskan
    if ENABLE_LLM and callable(ollama_chat):
        # susun prompt yang ringkas dan aman
        items = "\n".join([
            f"{i+1}. {r['nama_mobil']} ({r['tahun']}), harga {r['harga']}, "
            f"bahan bakar {r['bahan_bakar']}, transmisi {r['transmisi']}, kapasitas {r['kapasitas_mesin']}, usia {r['usia']} tahun"
            for i, r in enumerate(recs)
        ])
        prompt = (
            "Anda adalah asisten showroom mobil bekas. "
            "Ringkas rekomendasi untuk pengguna berdasarkan pertanyaan berikut dan daftar hasil yang sudah difilter. "
            "Hindari mengarang data baru, hanya gunakan yang diberikan. Maks 5 bullet.\n\n"
            f"Pertanyaan pengguna: {pertanyaan}\n\n"
            "Daftar hasil:\n" + items + "\n\n"
            "Tulis ringkasan yang menyorot kecocokan (bahan bakar, transmisi, harga), "
            "lalu beri saran singkat model mana yang paling relevan."
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
