from pathlib import Path
import os
import re
import math
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
PRICE_MARGIN = float(os.getenv("PRICE_MARGIN", "0.10"))

def _clean_name(nm: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(nm)).strip()

FUEL_KEYWORDS = {
    "listrik": ["listrik", "electric", "ev"],
    "hybrid":  ["hybrid", "hev", "phev", "plugin"],
    "diesel":  ["diesel"],
    "bensin":  ["bensin", "gasoline", "pertalite", "pertamax"],
}

BRANDS = ["bmw","toyota","daihatsu","wuling","hyundai","renault","honda",
          "suzuki","ford","mitsubishi","innova","fortuner","ayla","pajero","mobilio"]

def _as_rupiah(s: str) -> int:
    if not s:
        return 0
    s = s.lower().strip()
    m = re.match(r"([\d\.]+)\s*(jt|juta|jutaan)?", s)
    if not m:
        return int("".join(re.findall(r"\d+", s)) or "0")
    raw, unit = m.group(1), m.group(2)
    val = int(raw.replace(".", ""))
    if unit in ("jt", "juta", "jutaan") or val <= 10000:
        return val * 1_000_000
    return val

def _parse_query(q: str):
    ql = q.lower()
    parsed = {
        "brand": None, "fuel": None, "transmisi": None,
        "harga_min": None, "harga_max": None, "harga_target": None,
        "usia_max": None
    }
    for b in BRANDS:
        if b in ql: parsed["brand"] = b; break
    for key, keys in FUEL_KEYWORDS.items():
        if any(k in ql for k in keys): parsed["fuel"] = key; break
    if "matic" in ql or "otomatis" in ql:
        parsed["transmisi"] = "matic"
    elif "manual" in ql:
        parsed["transmisi"] = "manual"

    m = re.search(r"(?:di\s*bawah|<=|maks(?:imal)?|max)\s*([^\s]+(?:\s*(?:jt|juta|jutaan))?)", ql)
    if m: parsed["harga_max"] = _as_rupiah(m.group(1))
    m = re.search(r"(?:di\s*atas|lebih\s*dari|>=|min(?:imal)?)\s*([^\s]+(?:\s*(?:jt|juta|jutaan))?)", ql)
    if m: parsed["harga_min"] = _as_rupiah(m.group(1))
    m = re.search(r"(?:sekitar|kisaran|~|±)\s*([^\s]+(?:\s*(?:jt|juta|jutaan))?)", ql)
    if m: parsed["harga_target"] = _as_rupiah(m.group(1))
    if parsed["harga_min"] is None and parsed["harga_max"] is None and parsed["harga_target"] is None:
        m = re.search(r"(\d[\d\.]*)\s*(jt|juta|jutaan)?", ql)
        if m: parsed["harga_target"] = _as_rupiah(m.group(0))
    m = re.search(r"di\s*bawah\s*(\d+)\s*tahun", ql)
    if m: parsed["usia_max"] = int(m.group(1))
    return parsed

def _match_fuel_value(val: str, want: str) -> bool:
    if not want: return True
    s = str(val).lower()
    return any(k in s for k in FUEL_KEYWORDS.get(want, [want]))

def jawab_rule(pertanyaan: str, topk: int = 5):
    p = _parse_query(pertanyaan)
    out = df.copy()

    if p["brand"]:
        out = out[out["nama mobil"].str.contains(p["brand"], case=False, na=False)]
    if p["fuel"]:
        out = out[out["bahan bakar"].apply(lambda x: _match_fuel_value(x, p["fuel"]))]
    if p["transmisi"]:
        out = out[out["transmisi"].str.contains(p["transmisi"], case=False, na=False)]

    if p["harga_min"] is not None:
        out = out[out["harga_angka"] >= p["harga_min"]]
    if p["harga_max"] is not None:
        out = out[out["harga_angka"] <= p["harga_max"]]
    if out.empty:
        return []

    if p["usia_max"] is not None:
        out = out[out["usia"] <= p["usia_max"]]
        if out.empty: return []

    if p["harga_target"] is not None and p["harga_min"] is None and p["harga_max"] is None:
        target = p["harga_target"]
        relax_steps = [0.02, 0.05, 0.10, 0.20, 0.30]
        below = above = pd.DataFrame(columns=out.columns)

        for frac in relax_steps:
            lo = int(target * (1 - frac)); hi = int(target * (1 + frac))
            below = out[(out["harga_angka"] < target) & (out["harga_angka"] >= lo)]
            above = out[(out["harga_angka"] > target) & (out["harga_angka"] <= hi)]
            if not below.empty or not above.empty: break

        if below.empty: below = out[out["harga_angka"] < target]
        if above.empty: above = out[out["harga_angka"] > target]
        below, above = below.copy(), above.copy()
        below["diff"] = (target - below["harga_angka"]).abs()
        above["diff"] = (above["harga_angka"] - target).abs()
        below = below.sort_values(by=["diff", "usia"]).drop(columns=["diff"], errors="ignore")
        above = above.sort_values(by=["diff", "usia"]).drop(columns=["diff"], errors="ignore")

        n_below = (topk + 1) // 2
        n_above = topk - n_below
        pick_below = below.head(n_below)
        pick_above = above.head(n_above)

        if len(pick_below) < n_below:
            extra = above.iloc[n_above: n_above + (n_below - len(pick_below))]
            pick_above = pd.concat([pick_above, extra])
        if len(pick_above) < n_above:
            extra = below.iloc[n_below: n_below + (n_above - len(pick_above))]
            pick_below = pd.concat([pick_below, extra])

        out = pd.concat([pick_below, pick_above]).head(topk)

    kandidat_muda = out[out["usia"] <= PREFER_MAX_USIA]
    prefer = kandidat_muda if not kandidat_muda.empty else out
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

@router.get("")
def api_rule(pertanyaan: str = Query(..., description="Contoh: 'mobil listrik matic di bawah 500 juta' / 'mobil 500 juta'"),
             topk: int = Query(5, ge=1, le=50)):
    hasil = jawab_rule(pertanyaan, topk)
    if not hasil:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}
    lines = []
    for i, r in enumerate(hasil, 1):
        lines.append(
            f"{i}. {r['nama_mobil']} ({r['tahun']}) - {r['harga']} - "
            f"{r['bahan_bakar']}, {r['transmisi']}, {r['kapasitas_mesin']}"
        )
    return {"jawaban": "Hasil rekomendasi:\n\n" + "\n".join(lines), "rekomendasi": hasil}
