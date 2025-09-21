# app/embedding.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==== Konfigurasi ====
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "data_mobil_final.csv"

# Model embedding (kecil & cepat)
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
# Prioritas usia (tahun) – hasil dengan usia <= PREFER_MAX_USIA akan didahulukan
PREFER_MAX_USIA = int(os.getenv("PREFER_MAX_USIA", "5"))

# ==== Global state ====
_df: pd.DataFrame | None = None
_emb: np.ndarray | None = None
_model: SentenceTransformer | None = None


def _to_int(x, default=0):
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _to_price(s: str | int | float) -> str:
    """Normalisasi harga jadi 'Rp xxx.xxx.xxx' kalau memungkinkan."""
    try:
        n = int(float(str(s).replace(".", "").replace(",", "").replace("Rp", "").strip()))
        return f"Rp {n:,}".replace(",", ".")
    except Exception:
        return str(s)


def _build_search_text(row: pd.Series) -> str:
    """
    Gabungkan fitur penting menjadi satu kalimat untuk di-embed.
    Sesuaikan dengan kolom CSV kamu.
    """
    parts = []
    for col in ["nama_mobil", "bahan_bakar", "transmisi", "kapasitas_mesin", "tipe", "merk", "model"]:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    # tahun & harga ikut dimasukkan supaya query "500 juta" atau "2022" tetap nyantol
    if "tahun" in row and pd.notna(row["tahun"]):
        parts.append(f"tahun {row['tahun']}")
    if "harga" in row and pd.notna(row["harga"]):
        parts.append(f"harga {row['harga']}")
    return " | ".join(parts).lower()


def _ensure_loaded():
    global _df, _emb, _model
    if _df is not None and _emb is not None and _model is not None:
        return

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {DATA_CSV}")

    # Baca CSV
    df = pd.read_csv(DATA_CSV, dtype=str).fillna("")
    # Normalisasi kolom umum
    if "tahun" not in df.columns:
        df["tahun"] = ""
    if "harga" not in df.columns:
        df["harga"] = ""
    if "bahan_bakar" not in df.columns:
        df["bahan_bakar"] = ""
    if "transmisi" not in df.columns:
        df["transmisi"] = ""
    if "kapasitas_mesin" not in df.columns:
        df["kapasitas_mesin"] = ""
    if "nama_mobil" not in df.columns:
        # fallback kalau nama_mobil tak ada — pakai gabungan merk+model
        nm = []
        for _, r in df.iterrows():
            nm.append((r.get("nama") or r.get("model") or r.get("merk") or "Mobil").strip())
        df["nama_mobil"] = nm

    # Hitung usia (dinamis: pakai tahun sekarang)
    from datetime import datetime

    year_now = datetime.utcnow().year
    df["tahun_i"] = df["tahun"].apply(_to_int)
    df["usia"] = df["tahun_i"].apply(lambda y: max(0, year_now - y) if y > 0 else None)

    # Harga normalisasi tampilan
    df["harga_fmt"] = df["harga"].apply(_to_price)

    # Teks pencarian
    df["search_text"] = df.apply(_build_search_text, axis=1)

    # Siapkan model & embedding
    _model = SentenceTransformer(EMB_MODEL_NAME)
    emb = _model.encode(df["search_text"].tolist(), convert_to_numpy=True, show_progress_bar=False)

    # Simpan ke global
    _df = df
    _emb = emb


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity baris vektor A vs B (2D vs 2D)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def cosine_recommend(query: str, topk: int = 5) -> List[Dict[str, Any]]:
    """
    Cari rekomendasi dengan cosine similarity + prioritas usia <= PREFER_MAX_USIA.
    Return list of dict siap dipakai frontend.
    """
    _ensure_loaded()
    assert _df is not None and _emb is not None and _model is not None

    q_emb = _model.encode([query], convert_to_numpy=True)
    sims = _cosine_sim(q_emb, _emb)[0]  # (n_docs,)

    df = _df.copy()
    df["cosine_score"] = sims

    # Urutkan desc
    df = df.sort_values("cosine_score", ascending=False)

    # Prioritaskan usia <= PREFER_MAX_USIA (kalau kolom usia tersedia)
    young = df[df["usia"].apply(lambda x: x is not None and x <= PREFER_MAX_USIA)]
    old = df[df["usia"].apply(lambda x: x is None or x > PREFER_MAX_USIA)]

    # Ambil topk dengan prioritas
    rows = pd.concat([young, old]).head(topk)

    hasil = []
    for _, r in rows.iterrows():
        hasil.append(
            {
                "nama_mobil": r.get("nama_mobil", ""),
                "tahun": _to_int(r.get("tahun", 0)) or r.get("tahun", ""),
                "harga": r.get("harga_fmt") or r.get("harga", ""),
                "usia": r.get("usia"),
                "bahan_bakar": r.get("bahan_bakar", ""),
                "transmisi": r.get("transmisi", ""),
                "kapasitas_mesin": r.get("kapasitas_mesin", ""),
                "cosine_score": round(float(r.get("cosine_score", 0.0)), 4),
            }
        )
    return hasil
