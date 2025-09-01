from pathlib import Path
import pandas as pd
import re

APP_DIR = Path(__file__).resolve().parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"

data_mobil = pd.read_csv(DATA_CSV)
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

if 'harga_angka' not in data_mobil.columns:
    def bersihkan_harga(h):
        if pd.isna(h): return 0
        return int(re.sub(r'\D', '', str(h))) if re.search(r'\d', str(h)) else 0
    data_mobil['harga_angka'] = data_mobil['harga'].apply(bersihkan_harga)

def clean_name(nama):
    nama = str(nama).strip().lower()
    nama = re.sub(r'[^a-z0-9 ]', '', nama)
    nama = re.sub(r'\b(putih|merah|hitam|silver|abu|metalik|km|only|promo|limited|deluxe|std|double blower|special)\b', '', nama)
    nama = re.sub(r'\s+', ' ', nama)
    return nama.strip()

def jawab(pertanyaan: str):
    head_n = 316
    hasil = data_mobil.copy()
    q = pertanyaan.lower()

    if "listrik" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("listrik", case=False, na=False)]
    if "hybrid" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("hybrid", case=False, na=False)]

    if "manual" in q and "matic" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]
    elif "matic" in q and "manual" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]

    if "diesel" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("diesel", case=False, na=False)]
    if "bensin" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("bensin", case=False, na=False)]

    match = re.search(r'tahun (\d{4})\+', q)
    if match:
        hasil = hasil[hasil["tahun"] >= int(match.group(1))]

    usia_match = re.search(r'usia (?:di bawah|kurang dari) (\d+)', q)
    if usia_match:
        usia = int(usia_match.group(1))
        tahun_sekarang = 2025
        batas = tahun_sekarang - usia
        hasil = hasil[hasil["tahun"] >= batas]

    match = re.search(r"(?:di bawah|maximal|<=?) ?rp? ?(\d+[.\d]*)", q)
    if match:
        batas = int(match.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]

    output = "; ".join(
        clean_name(row["nama mobil"]) + f" ({row['tahun']})"
        for _, row in hasil.head(head_n).iterrows()
    )
    return output
