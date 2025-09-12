from pathlib import Path
import os
import re
import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/rule", tags=["Rekomendasi"])

APP_DIR = Path(__file__).resolve().parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

# ===== Load data =====
df = pd.read_csv(DATA_CSV)
df.columns = df.columns.str.strip().str.lower()

# normalisasi harga ke angka
if "harga_angka" not in df.columns:
    df["harga_angka"] = (
        df["harga"].astype(str).str.replace(r"[^\d]", "", regex=True).fillna("0").astype(int)
    )

PREFER_MAX_USIA = int(os.getenv("PREFER_MAX_USIA", "5"))

# ===== Helpers =====
def _clean_name(nm: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(nm)).strip()

FUEL_KEYWORDS = {
    "listrik": ["listrik", "electric", "ev"],
    "hybrid": ["hybrid", "hev", "phev", "plugin"],
    "diesel": ["diesel"],
    "bensin": ["bensin", "gasoline", "pertalite", "pertamax"]
}

BRANDS = [
    "bmw", "toyota", "daihatsu", "wuling", "hyundai",
    "renault", "honda", "suzuki", "ford", "mitsubishi",
    "innova", "fortuner", "ayla", "pajero", "mobilio"
]

def _parse_int(v: str) -> int:
    if not v:
        return 0
    # tangani 500jt / 500 juta / 500.000.000
    v = v.lower().replace("juta", "000000").replace("jt", "000000")
    return int("".join(re.findall(r"\d+", v)) or "0")

def _parse_query(q: str):
    ql = q.lower()

    parsed = {
        "brand": None,
        "fuel": None,          # normalized: listrik/hybrid/diesel/bensin
        "transmisi": None,     # matic/manual
        "harga_min": None,
        "harga_max": None,
        "usia_max": None       # jika user minta eksplisit
    }

    # brand
    for b in BRANDS:
        if b in ql:
            parsed["brand"] = b
            break

    # fuel
    for key, keys in FUEL_KEYWORDS.items():
        if any(k in ql for k in keys):
            parsed["fuel"] = key
            break

    # transmisi
    if "matic" in ql or "otomatis" in ql:
        parsed["transmisi"] = "matic"
    elif "manual" in ql:
        parsed["transmisi"] = "manual"

    # harga
    # di bawah / <=
    m = re.search(r"(?:di\s*bawah|<=|maks(?:imal)?|max)\s*([^\s]+(?:\s*(?:jt|juta))?)", ql)
    if m:
        parsed["harga_max"] = _parse_int(m.group(1))

    # di atas / >=
    m = re.search(r"(?:di\s*atas|lebih\s*dari|>=|min(?:imal)?)\s*([^\s]+(?:\s*(?:jt|juta))?)", ql)
    if m:
        parsed["harga_min"] = _parse_int(m.group(1))

    # angka tunggal (contoh: "500 juta") -> asumsikan max jika ada "bawah", kalau tidak biarkan
    if not parsed["harga_max"] and not parsed["harga_min"]:
        m = re.search(r"(\d[\d\.]*\s*(?:jt|juta)?)", ql)
        if m and ("bawah" in ql or "<" in ql):
            parsed["harga_max"] = _parse_int(m.group(1))

    # usia (contoh: di bawah 5 tahun)
    m = re.search(r"di\s*bawah\s*(\d+)\s*tahun", ql)
    if m:
        parsed["usia_max"] = int(m.group(1))

    return parsed

def _match_fuel(val: str, want: str) -> bool:
    if not want:
        return True
    s = str(val).lower()
    return any(k in s for k in FUEL_KEYWORDS.get(want, [want]))

# ===== Core =====
def jawab_rule(pertanyaan: str, topk: int = 5):
    p = _parse_query(pertanyaan)
    out = df.copy()

    # brand
    if p["brand"]:
        out = out[out["nama mobil"].str.contains(p["brand"], case=False, na=False)]

    # fuel
    if p["fuel"]:
        out = out[out["bahan bakar"].apply(lambda x: _match_fuel(x, p["fuel"]))]

    # transmisi
    if p["transmisi"]:
        out = out[out["transmisi"].str.contains(p["transmisi"], case=False, na=False)]

    # harga
    if p["harga_min"] is not None:
        out = out[out["harga_angka"] >= p["harga_min"]]
    if p["harga_max"] is not None:
        out = out[out["harga_angka"] <= p["harga_max"]]

    if out.empty:
        return []

    # jika user minta usia eksplisit -> filter ketat
    if p["usia_max"] is not None:
        out = out[out["usia"] <= p["usia_max"]]
        if out.empty:
            return []

    # prioritas usia <= PREFER_MAX_USIA
    kandidat_muda = out[out["usia"] <= PREFER_MAX_USIA]
    prefer = kandidat_muda if not kandidat_muda.empty else out

    # urutkan by harga lalu usia (lebih murah & muda di atas)
    prefer = prefer.sort_values(by=["harga_angka", "usia"], ascending=[True, True]).head(topk)

    hasil = []
    for _, r in prefer.iterrows():
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

# ===== API =====
@router.get("")
def api_rule(
    pertanyaan: str = Query(..., description="Contoh: 'mobil listrik matic di bawah 500 jt'"),
    topk: int = Query(5, ge=1, le=50)
):
    hasil = jawab_rule(pertanyaan, topk)
    if not hasil:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    lines = []
    for i, r in enumerate(hasil, 1):
        lines.append(f"{i}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - {r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}")
    return {"jawaban": "Hasil rekomendasi:\n\n" + "\n".join(lines), "rekomendasi": hasil}
