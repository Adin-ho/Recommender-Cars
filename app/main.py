# app/main.py
import os, traceback
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from embedding import cosine_recommend, DATA_CSV

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ChatCars")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "csv_exists": DATA_CSV.exists(),
        "csv_path": str(DATA_CSV),
        "base_dir": str(BASE_DIR),
    }

@app.get("/api/cosine_rekomendasi")
@app.get("/cosine_rekomendasi")  # alias fallback
def api_cosine(query: str = Query(..., min_length=2), k: int = 5):
    try:
        if not DATA_CSV.exists():
            return JSONResponse(
                {"ok": False, "error": f"CSV tidak ditemukan: {DATA_CSV}"},
                status_code=500,
            )
        rekom = cosine_recommend(query=query, topk=k)
        return {"ok": True, "rekomendasi": rekom}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"ok": False, "error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )
