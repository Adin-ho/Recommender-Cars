import pandas as pd
import re

data_mobil = pd.read_csv('app/data/data_mobil.csv')
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

if 'harga_angka' not in data_mobil.columns:
    def bersihkan_harga(h):
        if pd.isna(h): return 0
        return int(re.sub(r'\D', '', str(h))) if re.search(r'\d', str(h)) else 0
    data_mobil['harga_angka'] = data_mobil['harga'].apply(bersihkan_harga)

def jawab(pertanyaan: str):
    hasil = data_mobil.copy()
    q = pertanyaan.lower()
    tahun_sekarang = 2025

    # Usia
    if match := re.search(r'usia (?:di bawah|kurang dari) (\d+) tahun', q):
        usia = int(match.group(1))
        batas = tahun_sekarang - usia
        hasil = hasil[hasil["tahun"] >= batas]

    # Tahun
    if match := re.search(r'tahun (\d{4})\+', q):
        hasil = hasil[hasil["tahun"] >= int(match.group(1))]
    elif match := re.search(r'tahun (\d{4}) ke atas', q):
        hasil = hasil[hasil["tahun"] >= int(match.group(1))]
    elif match := re.search(r'tahun (?:di bawah|kurang dari) (\d{4})', q):
        hasil = hasil[hasil["tahun"] < int(match.group(1))]

    # Transmisi
    if "matic" in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
    if "manual" in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]

    # Bahan bakar
    for bb in ["diesel", "bensin", "hybrid", "listrik"]:
        if bb in q:
            hasil = hasil[hasil["bahan bakar"].str.contains(bb, case=False, na=False)]

    # Sinonim
    if "irit" in q or "hemat" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("bensin|hybrid", case=False, na=False)]

    # Harga
    if match := re.search(r"(?:di bawah|maximal|<=?) ?rp? ?(\d+[.\d]*)", q):
        batas = int(match.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]

    if hasil.empty:
        return ""

    def bersih_nama(nama, tahun):
        nama = nama.strip().lower()
        tahun = str(tahun).strip()
        nama = re.sub(r"\s*\(\d{4}\)$", "", nama)
        return f"{nama} ({tahun})"

    output = "; ".join(
        bersih_nama(row["nama mobil"], row["tahun"])
        for _, row in hasil.head(10).iterrows()
    )

    return output
