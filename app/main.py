import os
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========= PATHS =========
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"

# ========= APP =========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # longgar dulu biar mudah tes; ganti domain Anda kalau mau
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= DATA LOADING =========
if not DATA_CSV.exists():
    raise FileNotFoundError(f"Tidak ketemu dataset: {DATA_CSV}")

df = pd.read_csv(DATA_CSV)
# Normalisasi kolom
df.columns = df.columns.str.strip().str.lower()
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

# Pastikan nama kolom utama ada:
# nama_mobil, tahun, harga, usia, bahan_bakar, transmisi, kapasitas_mesin
def _ensure(col_name: str) -> None:
    if col_name not in df.columns:
        raise KeyError(f"Kolom '{col_name}' tidak ada di CSV")
for col in ["nama_mobil", "tahun", "harga", "bahan_bakar", "transmisi", "kapasitas_mesin"]:
    _ensure(col)

# harga -> angka
def to_int_price(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).lower().replace("rp", "").replace(" ", "")
    s = s.replace(",", ".")
    # format "150juta"
    m = re.search(r"(\d+)\s*juta", s)
    if m:
        return int(m.group(1)) * 1_000_000
    # ambil digit
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else 0

df["harga_angka"] = df["harga"].apply(to_int_price)

# Buat field teks untuk TF-IDF (gabung beberapa atribut biar “kaya”)
def row_text(r: pd.Series) -> str:
    parts = [
        str(r.get("nama_mobil", "")),
        str(r.get("bahan_bakar", "")),
        str(r.get("transmisi", "")),
        f"{str(r.get('kapasitas_mesin', ''))}cc",
        str(r.get("tahun", "")),
    ]
    return " ".join([p for p in parts if p])

df["_text"] = df.apply(row_text, axis=1)

# ========= TF-IDF FIT =========
# N-gram sampai bigram cukup, stop_words biarkan None (karena banyak istilah Indo)
VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
DOC_MATRIX = VECTORIZER.fit_transform(df["_text"].fillna(""))

# ========= UTIL PARSER =========
def parse_budget(q: str) -> int | None:
    """Cari 'di bawah 300 juta' atau angka langsung"""
    q = q.lower()
    # X juta
    m = re.search(r"(\d+)\s*juta", q)
    if m:
        return int(m.group(1)) * 1_000_000
    # angka penuh
    m2 = re.search(r"(\d[\d\.]{5,})", q)
    if m2:
        return int(m2.group(1).replace(".", ""))
    return None

def parse_tahun_min(q: str) -> int | None:
    m = re.search(r"tahun\s*(\d{4})\s*(ke\s*atas|keatas|>=)?", q.lower())
    if m:
        return int(m.group(1))
    return None

def parse_tahun_max(q: str) -> int | None:
    m = re.search(r"tahun\s*(?:di\s*bawah|<)\s*(\d{4})", q.lower())
    if m:
        return int(m.group(1))
    return None

def parse_usia_max(q: str) -> int | None:
    m = re.search(r"usia\s*(?:di\s*bawah|<)\s*(\d+)", q.lower())
    if m:
        return int(m.group(1))
    return None

def apply_rule_filters(base: pd.DataFrame, q: str) -> pd.DataFrame:
    ql = q.lower()
    d = base

    # bahan bakar
    fuels = ["diesel", "bensin", "hybrid", "listrik"]
    picked = [f for f in fuels if f in ql]
    if picked:
        regex = "|".join(picked)
        d = d[d["bahan_bakar"].astype(str).str.contains(regex, case=False, na=False)]

    # transmisi
    if "matic" in ql and "manual" not in ql:
        d = d[d["transmisi"].astype(str).str.contains("matic", case=False, na=False)]
    if "manual" in ql and "matic" not in ql:
        d = d[d["transmisi"].astype(str).str.contains("manual", case=False, na=False)]

    # budget
    budget = parse_budget(ql)
    if budget is not None:
        d = d[d["harga_angka"] <= budget]

    # tahun
    tmin = parse_tahun_min(ql)
    if tmin is not None:
        d = d[d["tahun"].astype(int) >= tmin]

    tmax = parse_tahun_max(ql)
    if tmax is not None:
        d = d[d["tahun"].astype(int) < tmax]

    # usia
    umax = parse_usia_max(ql)
    if umax is not None:
        tahun_sekarang = 2025  # boleh diset dinamis
        batas = tahun_sekarang - umax
        d = d[d["tahun"].astype(int) >= batas]

    return d

def rank_by_cosine(query: str, candidates_idx: np.ndarray, topk: int = 5) -> List[int]:
    """Hitung cosine similarity antara query dan semua dokumen, tapi hanya ambil indeks kandidat."""
    qvec = VECTORIZER.transform([query])
    scores = cosine_similarity(DOC_MATRIX[candidates_idx], qvec).ravel()
    ord_idx = np.argsort(scores)[::-1]  # descending
    take = ord_idx[:topk]
    return candidates_idx[take].tolist(), scores[take].tolist()

def to_payload(rows: pd.DataFrame, idxs: List[int], scores: List[float]) -> List[Dict[str, Any]]:
    out = []
    for i, s in zip(idxs, scores):
        r = rows.iloc[i]
        out.append({
            "nama_mobil": str(r.get("nama_mobil", "")),
            "tahun": int(r.get("tahun", 0)) if pd.notna(r.get("tahun")) else None,
            "harga": str(r.get("harga", "")),
            "usia": int(r.get("usia", 0)) if "usia" in rows.columns and pd.notna(r.get("usia")) else None,
            "bahan_bakar": str(r.get("bahan_bakar", "")),
            "transmisi": str(r.get("transmisi", "")),
            "kapasitas_mesin": str(r.get("kapasitas_mesin", "")),
            "cosine_score": float(round(s, 4)),
        })
    return out

# ========= ROUTES =========
@app.get("/health")
def health():
    return {"ok": True}

# Layani frontend
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(
    query: str = Query(..., description="Pertanyaan pengguna, mis. 'rekomendasi mobil listrik di bawah 300 juta'"),
    k: int = Query(5, ge=1, le=20),
):
    # 1) Filter rule-based lebih dulu
    filtered = apply_rule_filters(df, query)

    # Jika filter terlalu ketat, longgarkan (fallback ke semua data)
    if filtered.empty:
        filtered = df.copy()

    # 2) Siapkan kandidat index (index relatif terhadap df asli)
    candidates_idx = filtered.index.to_numpy()

    # 3) Rank dengan cosine TF-IDF
    top_idxs, scores = rank_by_cosine(query, candidates_idx, topk=k)

    if not top_idxs:
        return {"rekomendasi": []}

    payload = to_payload(df, top_idxs, scores)
    return {"rekomendasi": payload}
