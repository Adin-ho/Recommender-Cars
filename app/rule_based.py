# app/rule_based.py
from pathlib import Path
import pandas as pd
import re

_DF_CACHE = {"path": None, "df": None}

def _load_df(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if _DF_CACHE["df"] is None or _DF_CACHE["path"] != csv_path:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "harga_angka" not in df.columns and "harga" in df.columns:
            def to_int(x):
                s = re.sub(r"[^0-9]", "", str(x))
                return int(s) if s.isdigit() else 0
            df["harga_angka"] = df["harga"].apply(to_int)
        _DF_CACHE["path"], _DF_CACHE["df"] = csv_path, df
    return _DF_CACHE["df"]

def rekomendasi_rule_based(query: str, csv_path: Path, top_k: int = 5):
    df = _load_df(csv_path).copy()
    q = query.lower()

    # bahan bakar
    fuel_map = {"diesel": "diesel", "bensin": "bensin|pertamax|petrol", "listrik": "listrik|electric|ev"}
    for k, pat in fuel_map.items():
        if re.search(rf"\b{k}\b", q):
            df = df[df["bahan_bakar"].str.lower().str.contains(pat, na=False)]
            break

    # transmisi
    if re.search(r"\b(matic|otomatis|automatic)\b", q):
        df = df[df["transmisi"].str.lower().str.contains("matic|otomatis|automatic", na=False)]
    elif re.search(r"\bmanual\b", q):
        df = df[df["transmisi"].str.lower().str.contains("manual", na=False)]

    # harga (100-200 jt / <150 jt / >300 jt)
    if "harga_angka" in df.columns:
        m_range = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})\s*(jt|juta)", q)
        m_max   = re.search(r"(di\s*bawah|<)\s*(\d{2,3})\s*(jt|juta)", q)
        m_min   = re.search(r"(di\s*atas|>)\s*(\d{2,3})\s*(jt|juta)", q)
        if m_range:
            lo, hi = int(m_range.group(1))*1_000_000, int(m_range.group(2))*1_000_000
            df = df[(df["harga_angka"]>=lo) & (df["harga_angka"]<=hi)]
        elif m_max:
            hi = int(m_max.group(2))*1_000_000
            df = df[df["harga_angka"]<=hi]
        elif m_min:
            lo = int(m_min.group(2))*1_000_000
            df = df[df["harga_angka"]>=lo]

    # skor sederhana (match kata pada nama/merek)
    def score_row(r):
        text = f"{r.get('nama_mobil','')} {r.get('merek','')}".lower()
        return sum(tok in text for tok in re.findall(r"[a-z0-9]+", q))

    df["__score__"] = df.apply(score_row, axis=1)
    df = df.sort_values(["__score__", "harga_angka"], ascending=[False, True])

    cols = [c for c in ["nama_mobil","merek","tahun","transmisi","bahan_bakar","harga"] if c in df.columns]
    return df[cols].head(top_k).fillna("").to_dict(orient="records")
