import pandas as pd
import re
import csv
import os

# Load data mobil
mobil_df = pd.read_csv("../app/data/data_mobil.csv")
mobil_df.columns = mobil_df.columns.str.strip().str.lower()

# Load pertanyaan evaluasi
eval_path = "evaluasi_semua_batch.csv"
if not os.path.exists(eval_path):
    raise FileNotFoundError("❌ File evaluasi_semua_batch.csv tidak ditemukan!")

eval_df = pd.read_csv(eval_path)
if eval_df.empty or "pertanyaan" not in eval_df.columns:
    raise ValueError("❌ File evaluasi_semua_batch.csv kosong atau kolom 'pertanyaan' tidak ditemukan!")

def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())

mobil_df["nama mobil"] = mobil_df["nama mobil"].apply(normalize)
mobil_df["bahan bakar"] = mobil_df["bahan bakar"].apply(normalize)
mobil_df["transmisi"] = mobil_df["transmisi"].apply(normalize)
mobil_df["tahun"] = pd.to_numeric(
    mobil_df["tahun"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
).fillna(0).astype(int)
mobil_df["usia"] = pd.to_numeric(mobil_df["usia"], errors="coerce").fillna(0).astype(int)

def filter_mobil(pertanyaan):
    pertanyaan = normalize(pertanyaan)
    df = mobil_df.copy()

    if "manual" in pertanyaan:
        df = df[df["transmisi"] == "manual"]
    if "matic" in pertanyaan:
        df = df[df["transmisi"] == "matic"]

    for fuel in ["diesel", "bensin", "hybrid", "listrik"]:
        if fuel in pertanyaan:
            df = df[df["bahan bakar"] == fuel]

    tahun_match = re.search(r"(?:tahun|tahun produksi) (\d{4}) ke atas", pertanyaan)
    if tahun_match:
        min_tahun = int(tahun_match.group(1))
        df = df[df["tahun"] >= min_tahun]

    usia_match = re.search(r"(?:di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan)
    if usia_match:
        max_usia = int(usia_match.group(1))
        df = df[df["usia"] <= max_usia]

    harga_match = re.search(r"(?:di ?bawah|maksimal|<=?) ?rp? ?([\d\.]+)", pertanyaan)
    if harga_match:
        try:
            batas = int(harga_match.group(1).replace(".", ""))
            df["harga_angka"] = (
                df["harga"].astype(str)
                .str.replace("rp", "", case=False)
                .str.replace(".", "")
                .str.replace(",", "")
                .str.strip()
                .replace("", "0")
                .astype(float)
            )
            df = df[df["harga_angka"] <= batas]
        except:
            pass

    hasil = df.apply(lambda r: f"{r['nama mobil']} ({r['tahun']})", axis=1).tolist()
    return ";".join(sorted(set(hasil)))

# Proses dan simpan ulang
print("🔄 Menyusun ground truth dari dataset mobil...")
eval_df["ground_truth"] = eval_df["pertanyaan"].apply(filter_mobil)
eval_df.to_csv(eval_path, index=False, quoting=csv.QUOTE_ALL)
print("✅ Ground truth berhasil disimpan ulang.")
