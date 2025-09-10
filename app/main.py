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
    allow_origins=["*"],  # longgar biar mudah tes; ganti ke domain produksi kalau mau
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

# brand katalog (first token dari nama_mobil)
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
    # buang kata-kata umum
    stop = {"rekomendasi", "mobil", "tahun", "harga", "bawah", "di", "ke", "atas", "max", "maksimal", "rp", "jt", "juta"}
    cand = [t for t in tokens if t not in stop and len(t) >= 2]
    brands = [t for t in cand if t in KNOWN_BRANDS]
    return brands

# ========= FILTER & RANK =========
def apply_rule_filters(base: pd.DataFrame, q: str) -> pd.DataFrame:
    ql = q.lower()
    d = base

    # bahan bakar
    fuels = ["diesel", "bensin", "hybrid", "listrik", "ev", "electric"]
    picked = []
    for f in fuels:
        if re.search(fr"\b{re.escape(f)}\b", ql):
            picked.append("listrik" if f in {"ev", "electric"} else f)
    picked = list(dict.fromkeys(picked))  # unique preserve order
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
        d = d[d["harga_angka"] > 0]  # punya harga
        d = d[d["harga_angka"] <= budget]

    tmin = parse_tahun_min(ql)
    if tmin is not None:
        d = d[d["tahun"].astype(int) >= tmin]

    tmax = parse_tahun_max(ql)
    if tmax is not None:
        d = d[d["tahun"].astype(int) < tmax]

    umax = parse_usia_max(ql)
    if umax is not None:
        tahun_sekarang = 2025
        batas = tahun_sekarang - umax
        d = d[d["tahun"].astype(int) >= batas]

    return d

def rank_candidates(query: str, candidates_idx: np.ndarray, k: int, budget: int | None) -> tuple[list[int], list[float]]:
    """
    Skor gabungan:
      - cosine TF-IDF (w_text)
      - kedekatan ke budget (w_price) → kalau melewati budget, dipenalti
    """
    if len(candidates_idx) == 0:
        return [], []

    qvec = VECTORIZER.transform([query])
    cos = cosine_similarity(DOC_MATRIX[candidates_idx], qvec).ravel()

    # price proximity (0..1). Jika tidak ada budget → nol.
    price_prox = np.zeros_like(cos, dtype=float)
    if budget and budget > 0:
        prices = df.loc[candidates_idx, "harga_angka"].fillna(0).to_numpy(dtype=float)
        # kedekatan (boleh lebih kecil dari budget, yang melebihi dipenalti)
        prox = 1.0 - np.abs(prices - budget) / budget
        prox = np.clip(prox, 0.0, 1.0)
        # penalti jika > budget (turunkan 30%)
        over = prices > budget
        prox[over] *= 0.7
        price_prox = prox

    # bobot (bisa disetel)
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
        out.append({
            "nama_mobil": str(r.get("nama_mobil", "")),
            "tahun": int(r.get("tahun", 0)) if pd.notna(r.get("tahun")) else None,
            "harga": str(r.get("harga", "")),
            "usia": int(r.get("usia", 0)) if "usia" in rows.columns and pd.notna(r.get("usia")) else None,
            "bahan_bakar": str(r.get("bahan_bakar", "")),
            "transmisi": str(r.get("transmisi", "")),
            "kapasitas_mesin": str(r.get("kapasitas_mesin", "")),
            # frontend kamu membaca "cosine_score" → isi skor gabungan di sini
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

    # 1) filter rule-based (strict)
    filtered = apply_rule_filters(df, q)

    # 2) fallback kalau terlalu ketat
    if filtered.empty:
        filtered = df.copy()

    candidates_idx = filtered.index.to_numpy()
    budget = parse_budget(q)

    # 3) ranking gabungan
    top_idxs, scores = rank_candidates(q, candidates_idx, k, budget)

    if not top_idxs:
        return {"rekomendasi": []}

    payload = to_payload(df, top_idxs, scores)
    return {"rekomendasi": payload}
