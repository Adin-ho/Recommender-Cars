from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .embedding import embed_texts, embed_query

APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR / "data" / "data_mobil_final.csv"

# ====== Konfigurasi preferensi ======
PREFER_MAX_USIA = int(os.getenv("PREFER_MAX_USIA", "5")).__int__()

# ====== Load data sekali saja ======
_df = pd.read_csv(CSV_PATH)
_df.columns = _df.columns.str.strip().str.lower()

# normalisasi harga -> angka
if "harga_angka" not in _df.columns:
    _df["harga_angka"] = (
        _df["harga"].astype(str).str.replace(r"[^\d]", "", regex=True).fillna("0").astype(int)
    )

# kolom gabungan untuk semantic search
def _row_to_text(row: pd.Series) -> str:
    parts = [
        str(row.get("nama mobil", "")),
        str(row.get("bahan bakar", "")),
        str(row.get("transmisi", "")),
        str(row.get("kapasitas mesin", "")),
        str(row.get("tahun", "")),
        str(row.get("harga", "")),
    ]
    return " | ".join([p for p in parts if p and p != "nan"])

_corpus = _df.apply(_row_to_text, axis=1).tolist()
# Precompute embeddings korpus (sekali saat pertama dipakai)
_corpus_emb: np.ndarray | None = None


def _ensure_corpus_emb() -> np.ndarray:
    global _corpus_emb
    if _corpus_emb is None:
        _corpus_emb = embed_texts(_corpus)
    return _corpus_emb


# ====== Utility ======
_FUEL_KEYS = {
    "listrik": ["listrik", "electric", "ev"],
    "hybrid": ["hybrid", "hev", "phev", "plugin"],
    "diesel": ["diesel", "solar"],
    "bensin": ["bensin", "gasoline", "pertalite", "pertamax"],
}

def _fuel_match(val: str, want: str | None) -> bool:
    if not want:
        return True
    s = str(val).lower()
    keys = _FUEL_KEYS.get(want, [want])
    return any(k in s for k in keys)

def _parse_fuel_from_query(q: str) -> str | None:
    ql = q.lower()
    for k, keys in _FUEL_KEYS.items():
        if any(x in ql for x in keys):
            return k
    return None


# ====== Cosine rekomendasi ======
def cosine_rekomendasi(query: str, k: int = 5) -> Dict[str, Any]:
    """
    1) Hitung embedding query
    2) Cosine similarity ke korpus
    3) Ambil top-k, tetapi prioritaskan usia <= PREFER_MAX_USIA
    4) Jika user menyebut jenis bahan bakar, filter dulu
    """
    if not (query and query.strip()):
        return {"ok": False, "error": "Query kosong.", "rekomendasi": []}

    fuel_want = _parse_fuel_from_query(query)
    df = _df.copy()
    if fuel_want:
        df = df[df["bahan bakar"].apply(lambda x: _fuel_match(str(x), fuel_want))]
        if df.empty:
            return {"ok": True, "rekomendasi": []}

    # mapping index dataframe -> index korpus
    # (karena kita filter df, kita perlu subset embedding sesuai index)
    idx_keep = df.index.to_list()
    if not idx_keep:
        return {"ok": True, "rekomendasi": []}

    corpus_emb = _ensure_corpus_emb()[idx_keep]
    q_emb = embed_query(query)  # shape (D,)

    # cosine similarity -> vektor
    # (embeddings telah dinormalisasi; cukup dot product)
    scores = np.dot(corpus_emb, q_emb)  # shape (N,)
    df = df.copy()
    df["cosine_score"] = scores

    # Prioritaskan usia muda (<= PREFER_MAX_USIA), lalu skor, lalu harga naik
    muda = df[df["usia"] <= PREFER_MAX_USIA]
    tua = df[df["usia"] > PREFER_MAX_USIA]

    def _pick(d: pd.DataFrame, top: int) -> pd.DataFrame:
        if d.empty:
            return d
        return d.sort_values(by=["cosine_score", "harga_angka"], ascending=[False, True]).head(top)

    n_muda = min(len(muda), k)
    pick_muda = _pick(muda, n_muda)
    pick_tua = _pick(tua, max(0, k - n_muda))

    out = pd.concat([pick_muda, pick_tua]).head(k)

    hasil: List[Dict[str, Any]] = []
    for _, r in out.iterrows():
        hasil.append({
            "nama_mobil": str(r.get("nama mobil", "")),
            "tahun": int(r.get("tahun", 0)),
            "harga": r.get("harga", ""),
            "usia": int(r.get("usia", 0)),
            "bahan_bakar": r.get("bahan bakar", ""),
            "transmisi": r.get("transmisi", ""),
            "kapasitas_mesin": r.get("kapasitas mesin", ""),
            "cosine_score": round(float(r.get("cosine_score", 0.0)), 4),
        })

    return {"ok": True, "rekomendasi": hasil}
