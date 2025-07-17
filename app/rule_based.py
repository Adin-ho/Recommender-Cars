import pandas as pd
import re

# Load data
data_mobil = pd.read_csv('app/data/data_mobil_final.csv')
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

kunci_transmisi = ['matic', 'manual']
kunci_bahan_bakar = ['diesel', 'bensin', 'hybrid', 'listrik']
kunci_tanya = ['ada', 'apa', 'cari', 'butuh', 'ingin', 'saran', 'pilihan', 'mobil', 'mau', 'rekomendasi', 'tersedia']

def deteksi_kriteria(pertanyaan):
    # Normalisasi ke lower
    q = pertanyaan.lower()
    kriteria = {}

    # Transmisi
    for t in kunci_transmisi:
        if re.search(r'\b' + re.escape(t) + r'\b', q):
            kriteria['transmisi'] = t
    # Bahan bakar
    for b in kunci_bahan_bakar:
        if re.search(r'\b' + re.escape(b) + r'\b', q):
            kriteria['bahan_bakar'] = b
    # Tahun/harga/kapasitas (pattern angka)
    tahun = re.findall(r'(\d{4})', q)
    if tahun:
        kriteria['tahun'] = tahun
    harga = re.findall(r'(\d+\s?(juta|miliar|ribu)?)', q)
    if harga:
        kriteria['harga'] = harga
    # Deteksi kata tanya/aksi (agar tetap dilayani meski tanpa kata 'rekomendasi')
    for kt in kunci_tanya:
        if kt in q:
            kriteria['intent'] = kt
    # Merek umum (misal BMW, Toyota, Honda dst)
    merek_umum = ['bmw', 'toyota', 'honda', 'hyundai', 'chevrolet', 'ford', 'daihatsu', 'wuling', 'mercedes', 'nissan']
    for m in merek_umum:
        if m in q:
            kriteria['merek'] = m
    return kriteria

if 'harga_angka' not in data_mobil.columns:
    def bersihkan_harga(h):
        if pd.isna(h): return 0
        return int(re.sub(r'\D', '', str(h))) if re.search(r'\d', str(h)) else 0
    data_mobil['harga_angka'] = data_mobil['harga'].apply(bersihkan_harga)

def clean_name(nama):
    nama = str(nama).strip().lower()
    nama = re.sub(r'[^a-z0-9 ]', '', nama)
    # Hanya hilangkan warna, jangan hilangkan transmisi/varian!
    nama = re.sub(r'\b(putih|merah|hitam|silver|abu|metalik|km|only|promo|limited|deluxe|std|double blower|special)\b', '', nama)
    nama = re.sub(r'\s+', ' ', nama)
    return nama.strip()

def jawab(pertanyaan: str):
    head_n = 316
    hasil = data_mobil.copy()
    q = pertanyaan.lower()
    # 1. Filter Listrik
    if "listrik" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("listrik", case=False, na=False)]
    if "hybrid" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("hybrid", case=False, na=False)]
    # 2. Filter Transmisi
    if "manual" in q and "matic" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]
    elif "matic" in q and "manual" not in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
    # 3. Filter Diesel/Bensin
    if "diesel" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("diesel", case=False, na=False)]
    if "bensin" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("bensin", case=False, na=False)]
    # 4. Tahun
    match = re.search(r'tahun (\d{4})\+', q)
    if match:
        hasil = hasil[hasil["tahun"] >= int(match.group(1))]
    # 5. Usia
    usia_match = re.search(r'usia (?:di bawah|kurang dari) (\d+)', q)
    if usia_match:
        usia = int(usia_match.group(1))
        tahun_sekarang = 2025
        batas = tahun_sekarang - usia
        hasil = hasil[hasil["tahun"] >= batas]
    # 6. Harga
    match = re.search(r"(?:di bawah|maximal|<=?) ?rp? ?(\d+[.\d]*)", q)
    if match:
        batas = int(match.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]
    # Output
    output = "; ".join(
    clean_name(row["nama mobil"]) + f" ({row['tahun']})"
    for _, row in hasil.head(head_n).iterrows()
)
    return output
