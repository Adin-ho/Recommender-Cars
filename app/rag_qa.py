# app/main.py
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from typing import List, Dict, Any

# ==== IMPORT HELPER INTENT + SORT + PRETTY SCORE ====
from .rag_qa import detect_fuel_intent, pretty_scores, sort_young_first

app = FastAPI(title="Recommender Cars")

# === Static frontend ===
FRONT_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
FRONT_DIR = os.path.abspath(FRONT_DIR)
if os.path.isdir(FRONT_DIR):
    app.mount("/",
              StaticFiles(directory=FRONT_DIR, html=True),
              name="frontend")

# ---- Cari fungsi retrieval cosine dari embedding.py (apapun nama fungsinya) ----
def _load_engine():
    """
    Cari fungsi retrieval cosine yang ada di embedding.py dengan beberapa kemungkinan nama.
    Return: callable(query: str, topk: int) -> List[Dict]
    """
    from . import embedding  # module kamu sendiri
    candidates = [
        "engine_cosine_retrieve",
        "get_cosine_recommendations",
        "cosine_recommendations",
        "cosine_recommend",
        "cosine_search",
        "search_cosine",
        "retrieve_cosine",
    ]
    for name in candidates:
        fn = getattr(embedding, name, None)
        if callable(fn):
            return fn

    # Kalau tidak ada yang cocok, bikin raise yang jelas
    raise RuntimeError(
        "Tidak menemukan fungsi retrieval cosine di embedding.py. "
        "Harap ekspor salah satu nama fungsi ini: "
        + ", ".join(candidates)
    )

_engine_fn = None

def engine_cosine_retrieve(query: str, topk: int = 10) -> List[Dict[str, Any]]:
    global _engine_fn
    if _engine_fn is None:
        _engine_fn = _load_engine()
    return _engine_fn(query, topk=topk)

# ====== API ======

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/api/rule")
def api_rule(pertanyaan: str = Query(...), topk: int = 5):
    """
    Endpoint rule-based yang sudah kamu punya (biarkan sederhana).
    Kalau sebelumnya kamu sudah menulis di file lain, panggil dari sana.
    Di sini aku return kosong agar tidak mem-break existing.
    """
    try:
        from .rule_based import rule_answer
        data = rule_answer(pertanyaan, topk=topk)
        return data
    except Exception:
        return {"jawaban": "Maaf, tidak ada hasil dari rule-based.", "rekomendasi": []}

@app.get("/api/ask")
def api_ask(pertanyaan: str = Query(...), topk: int = 5):
    """
    1) Deteksi intent (diesel / listrik / hybrid).
    2) Ambil pool kandidat via cosine retrieval (topk besar).
    3) Filter ketat sesuai intent.
    4) Urut usia muda dulu lalu skor menurun.
    5) Normalisasi skor tampil (pretty_score) agar angka terlihat konsisten.
    """
    q = (pertanyaan or "").strip()
    intent = detect_fuel_intent(q)

    # Ambil pool lumayan besar biar filter tidak bikin kosong
    pool = engine_cosine_retrieve(q, topk=max(20, topk * 5))

    # Filter sesuai intent
    if intent == "diesel":
        pool = [x for x in pool if str(x.get("bahan_bakar", "")).lower().strip() == "diesel"]
    elif intent == "listrik":
        pool = [x for x in pool if "listrik" in str(x.get("bahan_bakar", "")).lower()]
    elif intent == "hybrid":
        pool = [x for x in pool if "hybrid" in str(x.get("bahan_bakar", "")).lower()]

    # Kalau kosong setelah filter, fallback: pakai top pool saja
    if not pool:
        pool = engine_cosine_retrieve(q, topk=max(20, topk * 3))

    # Urut usia muda dulu → skor desc
    sort_young_first(pool)
    # Normalisasi tampilan skor (pretty_score: 60..98/100 per-batch)
    pretty_scores(pool)

    # Ambil topk
    rekom = pool[:topk]

    return {
        "jawaban": "Rekomendasi berdasarkan Cosine Similarity:",
        "rekomendasi": rekom
    }

# (opsional) kembalikan index.html saat root dipanggil, kalau tidak pakai StaticFiles
@app.get("/index.html")
def index_page():
    path = os.path.join(FRONT_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"ok": True})
