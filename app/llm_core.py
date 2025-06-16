import pandas as pd
import re

# Load data
data_mobil = pd.read_csv("app/data/data_mobil.csv")
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

# Pastikan harga_angka tersedia
if "harga_angka" not in data_mobil.columns:
    def bersihkan_harga(h):
        if pd.isna(h): return 0
        return int(re.sub(r"\D", "", str(h))) if re.search(r"\d", str(h)) else 0
    data_mobil["harga_angka"] = data_mobil["harga"].apply(bersihkan_harga)

# Fungsi mirip rule, tapi anggap ini "LLM"
def jawab(pertanyaan: str) -> str:
    hasil = data_mobil.copy()
    q = pertanyaan.lower()

    # Filter dasar
    if "matic" in q: hasil = hasil[hasil["transmisi"].str.contains("matic", na=False, case=False)]
    if "manual" in q: hasil = hasil[hasil["transmisi"].str.contains("manual", na=False, case=False)]
    for bb in ["diesel", "bensin", "hybrid", "listrik"]:
        if bb in q:
            hasil = hasil[hasil["bahan bakar"].str.contains(bb, na=False, case=False)]

    # Harga
    match = re.search(r"(\d[\d\.]+)", q)
    if match:
        batas = int(match.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]

    # Tahun atas
    match = re.search(r"tahun (\d{4})\+?", q)
    if match:
        tahun_min = int(match.group(1))
        hasil = hasil[hasil["tahun"] >= tahun_min]

    # Usia (tahun sekarang - tahun mobil)
    if "usia di bawah" in q or "usia kurang dari" in q:
        usia = re.search(r"(?:usia di bawah|usia kurang dari) (\d+)", q)
        if usia:
            maks_usia = int(usia.group(1))
            hasil = hasil[2024 - hasil["tahun"] < maks_usia]

    # Format hasil
    if hasil.empty:
        return ""
    hasil = hasil.head(5)
    return "; ".join(f"{row['nama mobil'].strip().lower()} ({int(row['tahun'])})" for _, row in hasil.iterrows())
