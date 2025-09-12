from pathlib import Path
import re
import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/rule", tags=["Rule-based"])

APP_DIR = Path(__file__).resolve().parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

df = pd.read_csv(DATA_CSV)
df.columns = df.columns.str.strip().str.lower()

if "harga_angka" not in df.columns:
    df["harga_angka"] = df["harga"].str.replace(r"[^\d]", "", regex=True).astype(int)

def _clean_name(nm: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(nm)).strip()

def _parse_query(q: str):
    ql = q.lower()
    parsed = {
        "brand": None,
        "bahan_bakar": None,
        "transmisi": None,
        "harga_max": None
    }

    # brand
    brands = ["toyota", "daihatsu", "wuling", "bmw", "hyundai", "renault", "innova", "ayla", "fortuner", "mobilio"]
    for b in brands:
        if b in ql:
            parsed["brand"] = b
            break

    if "diesel" in ql:
        parsed["bahan_bakar"] = "diesel"
    elif "bensin" in ql:
        parsed["bahan_bakar"] = "bensin"

    if "matic" in ql or "otomatis" in ql:
        parsed["transmisi"] = "matic"
    elif "manual" in ql:
        parsed["transmisi"] = "manual"

    m = re.search(r"(?:di bawah|<=|maks(?:imal)?)\s*rp?\s*([\d\.]+)", ql)
    if m:
        parsed["harga_max"] = int(m.group(1).replace(".", ""))
    return parsed

def jawab_rule(pertanyaan: str, topk: int = 10):
    parsed = _parse_query(pertanyaan)
    out = df.copy()

    if parsed["brand"]:
        out = out[out["nama mobil"].str.contains(parsed["brand"], case=False, na=False)]

    if parsed["bahan_bakar"]:
        out = out[out["bahan bakar"].str.contains(parsed["bahan_bakar"], case=False, na=False)]

    if parsed["transmisi"]:
        out = out[out["transmisi"].str.contains(parsed["transmisi"], case=False, na=False)]

    if parsed["harga_max"]:
        out = out[out["harga_angka"] <= parsed["harga_max"]]

    if out.empty:
        return []

    out = out.sort_values("harga_angka").head(topk)

    hasil = []
    for _, r in out.iterrows():
        hasil.append({
            "nama_mobil": _clean_name(r.get("nama mobil", "")),
            "tahun": int(r.get("tahun", 0)),
            "harga": r.get("harga", ""),
            "usia": int(r.get("usia", 0)),
            "bahan_bakar": r.get("bahan bakar", ""),
            "transmisi": r.get("transmisi", ""),
            "kapasitas_mesin": r.get("kapasitas mesin", ""),
            "skor": None
        })
    return hasil

@router.get("")
def api_rule(pertanyaan: str = Query(..., description="Contoh: 'mobil diesel matic di bawah 300 juta'"), topk: int = 10):
    hasil = jawab_rule(pertanyaan, topk)
    if not hasil:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    lines = []
    for i, r in enumerate(hasil, 1):
        lines.append(f"{i}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - {r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}")
    return {"jawaban": "Hasil rule-based:\n\n" + "\n".join(lines), "rekomendasi": hasil}
