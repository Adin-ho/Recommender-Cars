# app/rule_based.py
from pathlib import Path
import re
import pandas as pd

# util kecil untuk lazy load CSV sekali saja
_DF_CACHE = {"path": None, "df": None}

def _load_df(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if _DF_CACHE["df"] is None or _DF_CACHE["path"] != csv_path:
        df = pd.read_csv(csv_path)
        # normalisasi kolom (aman kalau beda-beda)
        df.columns = [c.strip().lower() for c in df.columns]
        # siapkan kolom harga_angka bila belum ada
        if "harga_angka" not in df.columns and "harga" in df.columns:
            def to_int(x):
                s = str(x)
                s = re.sub(r"[^0-9]", "", s)
                return int(s) if s.isdigit() else 0
            df["harga_angka"] = df["harga"].apply(to_int)
        _DF_CACHE["path"], _DF_CACHE["df"] = csv_path, df
    return _DF_CACHE["df"]

def rekomendasi_rule_based(query: str, csv_path: Path, top_k: int = 5):
    """
    Filter sederhana berdasar kata kunci (bahan_bakar, transmisi, rentang harga, merek/nama).
    Return: list[dict]
    """
    df = _load_df(csv_path).copy()
    q = query.lower()

    # --- filter bahan bakar ---
    fuel_map = {"diesel": "diesel", "bensin": "bensin|pertamax|petrol|gasoline", "listrik": "listrik|electric|ev"}
    for k, pat in fuel_map.items():
        if re.search(rf"\b{k}\b", q):
            df = df[df["bahan_bakar"].str.lower().str.contains(pat, na=False)]
            break

    # --- filter transmisi ---
    if re.search(r"\b(matic|otomatis|automatic)\b", q):
        df = df[df["transmisi"].str.lower().str.contains("matic|otomatis|automatic", na=False)]
    elif re.search(r"\b(manual)\b", q):
        df = df[df["transmisi"].str.lower().str.contains("manual", na=False)]

    # --- filter harga (contoh: "< 200jt", "di bawah 150 juta", "100-200 juta") ---
    m_range = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})\s*(jt|juta)", q)
    m_max   = re.search(r"(di\s*bawah|<)\s*(\d{2,3})\s*(jt|juta)", q)
    m_min   = re.search(r"(di\s*atas|>)\s*(\d{2,3})\s*(jt|juta)", q)

    if "harga_angka" in df.columns:
        if m_range:
            lo, hi = int(m_range.group(1))*1_000_000, int(m_range.group(2))*1_000_000
            df = df[(df["harga_angka"] >= lo) & (df["harga_angka"] <= hi)]
        elif m_max:
            hi = int(m_max.group(2))*1_000_000
            df = df[df["harga_angka"] <= hi]
        elif m_min:
            lo = int(m_min.group(2))*1_000_000
            df = df[df["harga_angka"] >= lo]

    # --- skor sederhana: kemunculan kata pada nama/merek ---
    def score_row(row):
        text = f"{row.get('nama_mobil','')} {row.get('merek','')}".lower()
        score = 0
        for tok in re.findall(r"[a-z0-9]+", q):
            if tok in text:
                score += 1
        return score

    df["__score__"] = df.apply(score_row, axis=1)
    df = df.sort_values(["__score__", "harga_angka"], ascending=[False, True])
    out_cols = [c for c in ["nama_mobil","merek","tahun","transmisi","bahan_bakar","harga"] if c in df.columns]
    res = df[out_cols].head(top_k).fillna("").to_dict(orient="records")
    return res
