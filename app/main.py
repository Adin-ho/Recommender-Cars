import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========= PATHS =========
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"

# ========= APP =========
app = FastAPI(title="ChatCars")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti ke domain kamu jika mau lebih ketat
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= DATA =========
if not DATA_CSV.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_CSV}")

df = pd.read_csv(DATA_CSV)
df.columns = df.columns.str.strip().str.lower()
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

def _need(col: str):
    if col not in df.columns:
        raise KeyError(f"Kolom '{col}' tidak ada di CSV")
for c in ["nama_mobil", "tahun", "harga", "bahan_bakar", "transmisi", "kapasitas_mesin"]:
    _need(c)

# konversi harga → int (rupiah)
def to_int_price(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).lower().replace("rp", "").replace(" ", "")
    s = s.replace(",", ".")
    m = re.search(r"(\d+)\s*juta", s)
    if m:
        return int(m.group(1)) * 1_000_000
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else 0

df["harga_angka"] = df["harga"].apply(to_int_price)

# usia (thn sekarang 2025)
TAHUN_SEKARANG = 2025
df["_usia"] = (TAHUN_SEKARANG - df["tahun"].astype(int)).clip(lower=0)

# brand (token pertama)
def extract_brand(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0].lower()
df["_brand"] = df["nama_mobil"].astype(str).apply(extract_brand)
KNOWN_BRANDS = set(sorted(df["_brand"].unique()))

# field gabungan untuk TF-IDF
def row_text(r: pd.Series) -> str:
    parts = [
        str(r.get("nama_mobil", "")),
        str(r.get("bahan_bakar", "")),
        str(r.get("transmisi", "")),
        f"{str(r.get('kapasitas_mesin', ''))}",
        str(r.get("tahun", "")),
    ]
    return " ".join([p for p in parts if p])

df["_text"] = df.apply(row_text, axis=1).fillna("")

# ========= TF-IDF =========
VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
DOC_MATRIX = VECTORIZER.fit_transform(df["_text"])

# ========= PARSERS =========
def parse_budget(q: str) -> int | None:
    q = q.lower()
    m = re.search(r"(\d+)\s*juta", q)
    if m:
        return int(m.group(1)) * 1_000_000
    m2 = re.search(r"(\d[\d\.]{5,})", q)   # 500000000 / 500.000.000
    if m2:
        return int(m2.group(1).replace(".", ""))
    return None

def parse_tahun_min(q: str) -> int | None:
    m = re.search(r"tahun\s*(\d{4})\s*(ke\s*atas|keatas|>=)?", q.lower())
    return int(m.group(1)) if m else None

def parse_tahun_max(q: str) -> int | None:
    m = re.search(r"tahun\s*(?:di\s*bawah|<)\s*(\d{4})", q.lower())
    return int(m.group(1)) if m else None

def parse_usia_max(q: str) -> int | None:
    m = re.search(r"usia\s*(?:di\s*bawah|<)\s*(\d+)", q.lower())
    return int(m.group(1)) if m else None

def find_brand_tokens(q: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", q.lower())
    stop = {"rekomendasi", "mobil", "tahun", "harga", "bawah", "di", "ke", "atas", "max", "maksimal", "rp", "jt", "juta"}
    cand = [t for t in tokens if t not in stop and len(t) >= 2]
    brands = [t for t in cand if t in KNOWN_BRANDS]
    return brands

# ========= FILTER =========
def apply_rule_filters(base: pd.DataFrame, q: str) -> pd.DataFrame:
    ql = q.lower()
    d = base

    # bahan bakar
    fuels = ["diesel", "bensin", "hybrid", "listrik", "ev", "electric"]
    picked = []
    for f in fuels:
        if re.search(fr"\b{re.escape(f)}\b", ql):
            picked.append("listrik" if f in {"ev", "electric"} else f)
    picked = list(dict.fromkeys(picked))
    if picked:
        regex = "|".join(picked)
        d = d[d["bahan_bakar"].astype(str).str.contains(regex, case=False, na=False)]

    # transmisi
    if "matic" in ql and "manual" not in ql:
        d = d[d["transmisi"].astype(str).str.contains("matic", case=False, na=False)]
    if "manual" in ql and "matic" not in ql:
        d = d[d["transmisi"].astype(str).str.contains("manual", case=False, na=False)]

    # brand
    brands = find_brand_tokens(ql)
    if brands:
        d = d[d["_brand"].isin(brands)]

    # budget / tahun / usia
    budget = parse_budget(ql)
    if budget is not None:
        d = d[d["harga_angka"] > 0]
        d = d[d["harga_angka"] <= budget]

    tmin = parse_tahun_min(ql)
    if tmin is not None:
        d = d[d["tahun"].astype(int) >= tmin]

    tmax = parse_tahun_max(ql)
    if tmax is not None:
        d = d[d["tahun"].astype(int) < tmax]

    umax = parse_usia_max(ql)
    if umax is not None:
        batas = TAHUN_SEKARANG - umax
        d = d[d["tahun"].astype(int) >= batas]

    return d

# ========= RANKING =========
def rank_candidates(query: str, candidates_idx: np.ndarray, k: int, budget: int | None) -> tuple[list[int], list[float]]:
    if len(candidates_idx) == 0:
        return [], []

    qvec = VECTORIZER.transform([query])
    cos = cosine_similarity(DOC_MATRIX[candidates_idx], qvec).ravel()

    price_prox = np.zeros_like(cos, dtype=float)
    if budget and budget > 0:
        prices = df.loc[candidates_idx, "harga_angka"].fillna(0).to_numpy(dtype=float)
        prox = 1.0 - np.abs(prices - budget) / budget
        prox = np.clip(prox, 0.0, 1.0)
        over = prices > budget
        prox[over] *= 0.7  # penalti kalau lewat budget
        price_prox = prox

    w_text = 0.45
    w_price = 0.55 if budget else 0.0
    final = w_text * cos + w_price * price_prox

    order = np.argsort(final)[::-1]
    top = order[:k]
    return candidates_idx[top].tolist(), final[top].astype(float).tolist()

def to_payload(rows: pd.DataFrame, idxs: List[int], scores: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, s in zip(idxs, scores):
        r = rows.iloc[i]
        usia_val = int((TAHUN_SEKARANG - int(r.get("tahun", 0))) if pd.notna(r.get("tahun")) else 0)
        out.append({
            "nama_mobil": str(r.get("nama_mobil", "")),
            "tahun": int(r.get("tahun", 0)) if pd.notna(r.get("tahun")) else None,
            "harga": str(r.get("harga", "")),
            "usia": usia_val,
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

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(
    query: str = Query(..., description="mis. 'rekomendasi mobil listrik di bawah 300 juta'"),
    k: int = Query(5, ge=1, le=20),
):
    q = query.strip()
    if not q:
        return {"rekomendasi": []}

    # 1) filter aturan
    filtered = apply_rule_filters(df, q)
    if filtered.empty:
        filtered = df.copy()

    candidates_idx = filtered.index.to_numpy()
    budget = parse_budget(q)

    # 2) PRIORITAS USIA MUDA (≤ 5 tahun)
    usia_maks_muda = 5
    muda_mask = filtered["_usia"] <= usia_maks_muda
    idx_muda = filtered.index[muda_mask].to_numpy()
    idx_tua = filtered.index[~muda_mask].to_numpy()

    final_ids: List[int] = []
    final_scores: List[float] = []

    # 2a) ambil dari yang muda dulu
    ids_y, scores_y = rank_candidates(q, idx_muda, k, budget)
    final_ids.extend(ids_y)
    final_scores.extend(scores_y)

    # 2b) kalau belum cukup, isi dari yang >5 tahun
    if len(final_ids) < k and len(idx_tua) > 0:
        ids_o, scores_o = rank_candidates(q, idx_tua, k - len(final_ids), budget)
        final_ids.extend(ids_o)
        final_scores.extend(scores_o)

    if not final_ids:
        return {"rekomendasi": []}

    payload = to_payload(df, final_ids, final_scores)
    return {"rekomendasi": payload}
