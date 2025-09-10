from pathlib import Path
import re
import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/rule", tags=["Rule-based"])

APP_DIR = Path(__file__).resolve().parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

df = pd.read_csv(DATA_CSV)
df.columns = df.columns.str.strip().str.lower()

# helper: harga_angka
if "harga_angka" not in df.columns:
    def _to_int(h):
        s = str(h)
        m = re.findall(r"\d+", s)
        return int("".join(m)) if m else 0
    df["harga_angka"] = df["harga"].apply(_to_int)

def _clean_name(nm: str) -> str:
    nm = re.sub(r"\s*\(\d{4}\)$", "", str(nm)).strip()
    return nm

def jawab_rule(q: str, topk: int = 10) -> list[dict]:
    q_low = q.lower()
    out = df.copy()

    # bahan bakar
    if "diesel" in q_low: out = out[out["bahan bakar"].str.contains("diesel", case=False, na=False)]
    if "bensin" in q_low: out = out[out["bahan bakar"].str.contains("bensin|gasoline|petrol", case=False, na=False)]

    # transmisi
    if "matic" in q_low or " otomatis" in q_low: out = out[out["transmisi"].str.contains("matic|otomatis|auto", case=False, na=False)]
    if "manual" in q_low: out = out[out["transmisi"].str.contains("manual", case=False, na=False)]

    # usia
    m = re.search(r"usia\s*(?:maksimal|<=?|di bawah)?\s*(\d+)", q_low)
    if m:
        max_usia = int(m.group(1))
        out = out[out["usia"] <= max_usia]

    # kapasitas mesin (cc)
    m = re.search(r"(\d{3,4})\s*cc", q_low)
    if m:
        cc = int(m.group(1))
        out = out[out["kapasitas mesin"].astype(str).str.contains(str(cc))]

    # tahun minimal
    m = re.search(r"tahun\s*(?:>=|minimal|sejak)\s*(\d{4})", q_low)
    if m:
        th = int(m.group(1))
        out = out[out["tahun"] >= th]

    # harga batas
    m = re.search(r"(?:<=|di bawah|max|maks(?:imal)?)\s*rp?\s*([\d\.]+)", q_low)
    if m:
        batas = int(m.group(1).replace(".", ""))
        out = out[out["harga_angka"] <= batas]

    # sort by harga_angka naik sebagai default
    out = out.sort_values("harga_angka", ascending=True).head(topk)

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

@router.get("", summary="Jawab (rule-based)")
def api_rule(pertanyaan: str = Query(..., description="Contoh: 'mobil diesel matic di bawah 300 juta'"), topk: int = 10):
    recs = jawab_rule(pertanyaan, topk=topk)
    if not recs:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}
    # Buat ringkasan singkat
    lines = []
    for i, r in enumerate(recs, 1):
        lines.append(f"{i}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - {r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}")
    return {"jawaban": "Hasil rule-based:\n\n" + "\n".join(lines), "rekomendasi": recs}
