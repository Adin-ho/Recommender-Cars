from pathlib import Path
import re
import pandas as pd
from fastapi import APIRouter, Query
import os

router = APIRouter(prefix="/api/rule", tags=["Rule-based"])

APP_DIR = Path(__file__).resolve().parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

# ==== Load data ====
df = pd.read_csv(DATA_CSV)
# Normalisasi kolom
df.columns = df.columns.str.strip().str.lower()

# harga_angka
if "harga_angka" not in df.columns:
    def _to_int_idr(x):
        s = str(x)
        digits = re.findall(r"\d+", s)
        return int("".join(digits)) if digits else 0
    df["harga_angka"] = df["harga"].apply(_to_int_idr)

# bantu: bersihkan nama
def _clean_name(nm: str) -> str:
    nm = str(nm)
    # hapus akhiran " (YYYY)"
    nm = re.sub(r"\s*\(\d{4}\)$", "", nm).strip()
    # hapus dobel spasi
    nm = re.sub(r"\s{2,}", " ", nm)
    return nm

# preferensi usia (default 5 tahun)
PREFER_MAX_USIA = int(os.getenv("PREFER_MAX_USIA", "5"))

# ==== Parsing kueri ====
def _parse_query(q: str) -> dict:
    ql = q.lower()
    out = {
        "bahan_bakar": None,   # "diesel"|"bensin"|None
        "transmisi": None,     # "matic"|"manual"|None
        "usia_max": None,      # int|None
        "tahun_min": None,     # int|None
        "harga_max": None,     # int (rupiah)|None
    }

    # bahan bakar
    if "diesel" in ql: out["bahan_bakar"] = "diesel"
    elif "bensin" in ql or "petrol" in ql or "gasoline" in ql: out["bahan_bakar"] = "bensin"

    # transmisi
    if "matic" in ql or "otomatis" in ql or "automatic" in ql: out["transmisi"] = "matic"
    elif "manual" in ql: out["transmisi"] = "manual"

    # pola usia (mis. "≤ 5 tahun", "5 tahun ke bawah/di bawah")
    m = re.search(r"(?:usia\s*)?(?:<=|≤|di bawah|ke\s*bawah|maks(?:imal)?)\s*(\d{1,2})\s*tahun", ql)
    if m: out["usia_max"] = int(m.group(1))

    # pola "tahun minimal 2019", "sejak 2020"
    m = re.search(r"(?:tahun\s*(?:minimal|>=|sejak)\s*)(\d{4})", ql)
    if m: out["tahun_min"] = int(m.group(1))

    # harga max: "di bawah 300 juta" / "<= 250.000.000"
    m = re.search(r"(?:<=|di bawah|maks(?:imal)?)\s*([0-9\.]+)\s*(juta|jt|milyar|miliar|m)?", ql)
    if m:
        val = m.group(1).replace(".", "")
        unit = m.group(2) or ""
        n = int(val)
        if unit in ("juta", "jt", "m"): n *= 1_000_000
        elif unit in ("milyar", "miliar"): n *= 1_000_000_000
        out["harga_max"] = n
    else:
        # fallback: angka rupiah mentah
        m2 = re.search(r"rp?\s*([0-9\.]{6,})", ql)
        if m2:
            out["harga_max"] = int(m2.group(1).replace(".", ""))

    return out

def _apply_filters(frame: pd.DataFrame, parsed: dict) -> pd.DataFrame:
    out = frame.copy()

    if parsed["bahan_bakar"] == "diesel":
        out = out[out["bahan bakar"].str.contains("diesel", case=False, na=False)]
    elif parsed["bahan_bakar"] == "bensin":
        out = out[out["bahan bakar"].str.contains("bensin|petrol|gasoline", case=False, na=False)]

    if parsed["transmisi"] == "matic":
        out = out[out["transmisi"].str.contains("matic|otomatis|auto", case=False, na=False)]
    elif parsed["transmisi"] == "manual":
        out = out[out["transmisi"].str.contains("manual", case=False, na=False)]

    if parsed["usia_max"] is not None and "usia" in out.columns:
        out = out[out["usia"] <= parsed["usia_max"]]

    if parsed["tahun_min"] is not None and "tahun" in out.columns:
        out = out[out["tahun"] >= parsed["tahun_min"]]

    if parsed["harga_max"] is not None:
        out = out[out["harga_angka"] <= parsed["harga_max"]]

    return out

def _sort_with_preference(frame: pd.DataFrame) -> pd.DataFrame:
    # prioritas usia <= PREFER_MAX_USIA, lalu termurah, lalu termuda
    prefer_mask = (frame["usia"] <= PREFER_MAX_USIA)
    return (
        frame.assign(_prefer=prefer_mask.astype(int))
             .sort_values(by=["_prefer", "harga_angka", "usia"], ascending=[False, True, True])
             .drop(columns=["_prefer"])
    )

def jawab_rule(q: str, topk: int = 10) -> list[dict]:
    parsed = _parse_query(q)
    cand = _apply_filters(df, parsed)
    if cand.empty:
        # tidak ada hasil setelah filter → pakai semua data tapi tetap prioritaskan usia muda
        cand = df.copy()

    cand = _sort_with_preference(cand).head(topk)

    hasil = []
    for _, r in cand.iterrows():
        hasil.append({
            "nama_mobil": _clean_name(r.get("nama mobil", "")),
            "tahun": int(r.get("tahun", 0)) if pd.notna(r.get("tahun", None)) else None,
            "harga": r.get("harga", ""),
            "usia": int(r.get("usia", 0)) if pd.notna(r.get("usia", None)) else None,
            "bahan_bakar": r.get("bahan bakar", ""),
            "transmisi": r.get("transmisi", ""),
            "kapasitas_mesin": r.get("kapasitas mesin", ""),
            "skor": None
        })
    return hasil

@router.get("", summary="Jawab (rule-based)")
def api_rule(
    pertanyaan: str = Query(..., description="Contoh: 'mobil diesel matic di bawah 300 juta'"),
    topk: int = 10
):
    recs = jawab_rule(pertanyaan, topk=topk)
    if not recs:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    # ringkas untuk tampilan
    lines = []
    for i, r in enumerate(recs, 1):
        tahun = f" ({r['tahun']})" if r["tahun"] else ""
        lines.append(f"{i}. {r['nama_mobil']}{tahun} - {r['harga']} - {r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}")
    return {"jawaban": "Hasil rule-based:\n\n" + "\n".join(lines), "rekomendasi": recs}
