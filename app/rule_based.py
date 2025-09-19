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
PRICE_MARGIN = float(os.getenv("PRICE_MARGIN", "0.10"))  # 10% jendela awal untuk "harga sekitar"

# ===== Helpers =====
def _clean_name(nm: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(nm)).strip()

FUEL_KEYWORDS = {
    "listrik": ["listrik", "electric", "ev"],
    "hybrid":  ["hybrid", "hev", "phev", "plugin"],
    "diesel":  ["diesel"],
    "bensin":  ["bensin", "gasoline", "pertalite", "pertamax"]
}

BRANDS = [
    "bmw", "toyota", "daihatsu", "wuling", "hyundai",
    "renault", "honda", "suzuki", "ford", "mitsubishi",
    "innova", "fortuner", "ayla", "pajero", "mobilio"
]

def _as_rupiah(s: str) -> int:
    """Parse angka + satuan (jt/juta). Jika tanpa satuan dan angka kecil (<= 10.000), diasumsikan 'juta'."""
    if not s:
        return 0
    s = s.lower().strip()
    m = re.match(r"([\d\.]+)\s*(jt|juta|jutaan)?", s)
    if not m:
        # fallback: ambil semua digit yang ada
        num = int("".join(re.findall(r"\d+", s)) or "0")
        return num
    raw, unit = m.group(1), m.group(2)
    val = int(raw.replace(".", ""))
    if unit in ("jt", "juta", "jutaan") or val <= 10000:
        # 500 -> 500 juta
        return val * 1_000_000
    return val

def _parse_query(q: str):
    ql = q.lower()

    parsed = {
        "brand": None,
        "fuel": None,          # normalized: listrik/hybrid/diesel/bensin
        "transmisi": None,     # matic/manual
        "harga_min": None,
        "harga_max": None,
        "harga_target": None,  # <-- baru: untuk "sekitar 500 juta" atau angka polos
        "usia_max": None
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

    # harga <= / di bawah
    m = re.search(r"(?:di\s*bawah|<=|maks(?:imal)?|max)\s*([^\s]+(?:\s*(?:jt|juta|jutaan))?)", ql)
    if m:
        parsed["harga_max"] = _as_rupiah(m.group(1))

    # harga >= / di atas
    m = re.search(r"(?:di\s*atas|lebih\s*dari|>=|min(?:imal)?)\s*([^\s]+(?:\s*(?:jt|juta|jutaan))?)", ql)
    if m:
        parsed["harga_min"] = _as_rupiah(m.group(1))

    # kata kunci "sekitar/kisaran/±" -> harga_target
    m = re.search(r"(?:sekitar|kisaran|~|±)\s*([^\s]+(?:\s*(?:jt|juta|jutaan))?)", ql)
    if m:
        parsed["harga_target"] = _as_rupiah(m.group(1))

    # jika belum ada min/max/target dan ada angka polos -> anggap target
    if (parsed["harga_min"] is None and parsed["harga_max"] is None and parsed["harga_target"] is None):
        m = re.search(r"(\d[\d\.]*)\s*(jt|juta|jutaan)?", ql)
        if m:
            parsed["harga_target"] = _as_rupiah(m.group(0))

    # usia (contoh: di bawah 5 tahun)
    m = re.search(r"di\s*bawah\s*(\d+)\s*tahun", ql)
    if m:
        parsed["usia_max"] = int(m.group(1))

    return parsed

def _match_fuel_value(val: str, want: str) -> bool:
    """Cek 'bahan bakar' terhadap sinonim (listrik/electric/ev, hybrid/hev/phev, dst)."""
    if not want:
        return True
    s = str(val).lower()
    return any(k in s for k in FUEL_KEYWORDS.get(want, [want]))

# ===== Core =====
def jawab_rule(pertanyaan: str, topk: int = 5):
    p = _parse_query(pertanyaan)
    out = df.copy()

    # BRAND
    if p["brand"]:
        out = out[out["nama mobil"].str.contains(p["brand"], case=False, na=False)]

    # FUEL
    if p["fuel"]:
        out = out[out["bahan bakar"].apply(lambda x: _match_fuel_value(x, p["fuel"]))]

    # TRANSMISI
    if p["transmisi"]:
        out = out[out["transmisi"].str.contains(p["transmisi"], case=False, na=False)]

    # HARGA min/max
    if p["harga_min"] is not None:
        out = out[out["harga_angka"] >= p["harga_min"]]
    if p["harga_max"] is not None:
        out = out[out["harga_angka"] <= p["harga_max"]]

    if out.empty:
        return []

    # USIA eksplisit dari user -> filter ketat
    if p["usia_max"] is not None:
        out = out[out["usia"] <= p["usia_max"]]
        if out.empty:
            return []

    # === PRICE-ANCHOR: angka polos/sekitar 500 jt -> cari sekitar target ===
    if p["harga_target"] is not None and p["harga_min"] is None and p["harga_max"] is None:
        target = p["harga_target"]
        def _window(frac: float):
            lo = int(target * (1.0 - frac))
            hi = int(target * (1.0 + frac))
            return out[(out["harga_angka"] >= lo) & (out["harga_angka"] <= hi)]

        # coba ±10% → ±20% → ±30%
        cand = _window(PRICE_MARGIN)
        if cand.empty:
            cand = _window(max(PRICE_MARGIN * 2, 0.20))
        if cand.empty:
            cand = _window(max(PRICE_MARGIN * 3, 0.30))

        if cand.empty:
            # fallback: pilih yang terdekat terhadap target
            cand = out.copy()
            cand["diff_abs"] = (cand["harga_angka"] - target).abs()
            cand = cand.sort_values(by=["diff_abs", "usia"], ascending=[True, True]).head(topk)
        else:
            # prioritas: paling dekat ke target & lebih muda
            cand["diff_abs"] = (cand["harga_angka"] - target).abs()
            cand = cand.sort_values(by=["diff_abs", "usia"], ascending=[True, True]).head(topk)

        out = cand.drop(columns=[c for c in ["diff_abs"] if c in cand.columns])

    # PRIORITAS USIA <= PREFER_MAX_USIA, jika tidak ada pakai semua
    kandidat_muda = out[out["usia"] <= PREFER_MAX_USIA]
    prefer = kandidat_muda if not kandidat_muda.empty else out

    # URUT default: termurah & termuda
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
    pertanyaan: str = Query(..., description="Contoh: 'mobil listrik matic di bawah 500 juta' / 'mobil 500 juta'"),
    topk: int = Query(5, ge=1, le=50)
):
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
