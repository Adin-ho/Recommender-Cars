import pandas as pd
import re
import csv

# Load dataset mobil
mobil_df = pd.read_csv("app/data/data_mobil.csv")
mobil_df.columns = mobil_df.columns.str.strip().str.lower()

# Load pertanyaan evaluasi
eval_path = "evaluation/evaluasi_semua_batch.csv"
eval_df = pd.read_csv(eval_path)

def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())

# Normalisasi kolom
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

    # Filter by transmisi
    if "manual" in pertanyaan:
        df = df[df["transmisi"] == "manual"]
    if "matic" in pertanyaan:
        df = df[df["transmisi"] == "matic"]

    # Filter by bahan bakar
    for fuel in ["diesel", "bensin", "hybrid", "listrik"]:
        if fuel in pertanyaan:
            df = df[df["bahan bakar"] == fuel]

    # Filter by tahun
    tahun_min = None
    m = re.search(r"(?:tahun|tahun produksi) (\d{4})\+?", pertanyaan)
    if m:
        tahun_min = int(m.group(1))
        df = df[df["tahun"] >= tahun_min]

    # Filter by usia
    usia_max = None
    m = re.search(r"(?:di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan)
    if m:
        usia_max = int(m.group(1))
        df = df[df["usia"] <= usia_max]

    # Filter by harga
    harga_max = None
    m = re.search(r"(?:di ?bawah|maksimal|<=?) ?rp? ?([\d\.]+)", pertanyaan)
    if m:
        harga_max = int(m.group(1).replace(".", ""))
        df["harga_angka"] = (
            df["harga"].astype(str)
            .str.replace("rp", "", case=False)
            .str.replace(".", "")
            .str.replace(",", "")
            .str.strip()
            .replace("", "0")
            .astype(float)
        )
        df = df[df["harga_angka"] <= harga_max]

    # Buat ground truth nama mobil + tahun
    hasil = df.apply(lambda r: f"{r['nama mobil']} ({r['tahun']})", axis=1).tolist()
    return ";".join(sorted(set(hasil))) if hasil else ""

# Proses dan simpan ulang
print("🔄 Generate ground truth dari dataset mobil...")
eval_df["ground_truth"] = eval_df["pertanyaan"].apply(filter_mobil)
eval_df.to_csv(eval_path, index=False, quoting=csv.QUOTE_ALL)
print("✅ Berhasil memperbarui evaluasi_semua_batch.csv dengan ground truth valid (nama mobil + tahun).")
