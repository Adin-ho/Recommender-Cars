from __future__ import annotations
import os, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .embedding import load_dataset, load_embeddings, query_topk, build_and_persist_embeddings

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# ====== ENV ======
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
PREFER_MAX_USIA = int(os.getenv("PREFER_MAX_USIA", "5"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve UI
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/ping")
def ping():
    return {"ok": True}

@app.get("/")
def root():
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return HTMLResponse("<h1>OK</h1><p>Backend up. Open /docs untuk test API.</p>")

# ====== Data & index ======
_df: pd.DataFrame | None = None
_vectors = None
_ids: List[int] | None = None

def _ensure_loaded():
    global _df, _vectors, _ids
    if _df is None:
        _df = load_dataset()
        _df.columns = _df.columns.str.strip().str.lower()

        # harga_angka
        if "harga_angka" not in _df.columns:
            def to_int(s):
                import re
                if pd.isna(s): return 0
                s = str(s)
                m = re.findall(r"\d+", s)
                return int("".join(m)) if m else 0
            _df["harga_angka"] = _df["harga"].apply(to_int)

        # usia
        from datetime import datetime
        tahun_now = datetime.now().year
        _df["usia"] = _df["tahun"].apply(lambda t: max(0, tahun_now - int(t)))

    if _vectors is None:
        _vectors, _ids = load_embeddings()

# ====== Helpers ======
def _rule_filters(query: str, df: pd.DataFrame) -> pd.DataFrame:
    q = query.lower()

    fuels = ["diesel", "listrik", "hybrid", "bensin"]
    for f in fuels:
        if f in q:
            df = df[df["bahan bakar"].str.contains(f, case=False, na=False)]

    brands = ["toyota","honda","daihatsu","mitsubishi","wuling","bmw","mercedes","mazda","suzuki","nissan","hyundai","kia"]
    for b in brands:
        if re.search(rf"\b{re.escape(b)}\b", q):
            df = df[df["nama mobil"].str.contains(b, case=False, na=False)]

    # harga di bawah / max
    m = re.search(r"(?:di\s*bawah|max|<=?)\s*rp?\s*([\d\.]+)", q)
    if m:
        lim = int(m.group(1).replace(".", ""))
        df = df[df["harga_angka"] <= lim]

    # tahun X ke atas
    m2 = re.search(r"tahun\s*(\d{4})\s*ke\s*atas", q)
    if m2:
        df = df[df["tahun"] >= int(m2.group(1))]

    return df

def _prefer_young_first(rows: List[Tuple[int,float]], df: pd.DataFrame, k: int) -> List[Tuple[int,float]]:
    young, other = [], []
    for idx, sc in rows:
        usia = int(df.iloc[idx]["usia"])
        (young if usia <= PREFER_MAX_USIA else other).append((idx, sc))
    out = young[:k]
    if len(out) < k:
        out += other[:k-len(out)]
    return out

def _pack_row(row: pd.Series, score: float) -> Dict[str, Any]:
    return {
        "nama_mobil": row.get("nama mobil", ""),
        "tahun": int(row.get("tahun", 0)),
        "harga": row.get("harga", ""),
        "usia": int(row.get("usia", 0)),
        "bahan_bakar": row.get("bahan bakar", ""),
        "transmisi": row.get("transmisi", ""),
        "kapasitas_mesin": row.get("kapasitas mesin", ""),
        "cosine_score": round(float(score), 4)
    }

# ====== Endpoint handler (GET/POST) ======
def _handle_rekomendasi(query: str, k: int = 5):
    _ensure_loaded()
    if not query:
        return JSONResponse({"rekomendasi": []})

    df_f = _rule_filters(query, _df)
    if df_f.empty:
        df_f = _df

    idx_map = set(df_f.index.to_list())
    top_global = query_topk(_df, _vectors, query, k=max(k*5, 20))
    filtered = [(i, sc) for (i, sc) in top_global if i in idx_map] or top_global

    chosen = _prefer_young_first(filtered, _df, k)
    out = [_pack_row(_df.iloc[idx], sc) for idx, sc in chosen]
    return JSONResponse({"rekomendasi": out})

# GET versi lama
@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(query: str, k: int = 5):
    return _handle_rekomendasi(query, k)

# Mirror di /api/...
@app.get("/api/cosine_rekomendasi")
def cosine_rekomendasi_api(query: str, k: int = 5):
    return _handle_rekomendasi(query, k)

# POST dukungan (body JSON: {query,k})
@app.post("/api/cosine_rekomendasi")
async def cosine_rekomendasi_post(req: Request):
    data = await req.json()
    query = data.get("query", "")
    k = int(data.get("k", 5))
    return _handle_rekomendasi(query, k)

# Rebuild index embeddings
@app.post("/admin/rebuild_embeddings")
def rebuild_embeddings():
    try:
        build_and_persist_embeddings()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
