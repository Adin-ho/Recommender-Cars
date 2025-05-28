import pandas as pd
import re
import csv  # ✅ Tambahkan ini untuk QUOTE_ALL

# Load dataset mobil
mobil_df = pd.read_csv("../app/data/data_mobil.csv")
mobil_df.columns = mobil_df.columns.str.strip().str.lower()

# Load pertanyaan evaluasi
eval_df = pd.read_csv("evaluasi_semua_batch.csv")

def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())

# Normalisasi kolom utama
mobil_df["nama mobil"] = mobil_df["nama mobil"].apply(normalize)
mobil_df["bahan bakar"] = mobil_df["bahan bakar"].apply(normalize)
mobil_df["transmisi"] = mobil_df["transmisi"].apply(normalize)

# Tangani tahun dan usia
mobil_df["tahun"] = mobil_df["tahun"].astype(str).str.extract(r"(\d{4})")
mobil_df["tahun"] = pd.to_numeric(mobil_df["tahun"], errors="coerce").fillna(0).astype(int)
mobil_df["usia"] = pd.to_numeric(mobil_df["usia"], errors="coerce").fillna(0).astype(int)

# Fungsi untuk filter berdasarkan isi pertanyaan
def filter_mobil(pertanyaan):
    pertanyaan = normalize(pertanyaan)
    query = mobil_df.copy()

    if "manual" in pertanyaan:
        query = query[query["transmisi"] == "manual"]
    if "matic" in pertanyaan:
        query = query[query["transmisi"] == "matic"]

    for bahan in ["diesel", "bensin", "hybrid", "listrik"]:
        if bahan in pertanyaan:
            query = query[query["bahan bakar"] == bahan]

    match_tahun = re.search(r"(?:tahun|tahun produksi) (\d{4}) ke atas", pertanyaan)
    if match_tahun:
        tahun_min = int(match_tahun.group(1))
        query = query[query["tahun"] >= tahun_min]

    match_usia = re.search(r"(?:di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan)
    if match_usia:
        usia_max = int(match_usia.group(1))
        query = query[query["usia"] <= usia_max]

    match_harga = re.search(r"(?:di ?bawah|maksimal|<=?) ?rp? ?([\d\.]+)", pertanyaan)
    if match_harga:
        harga_str = match_harga.group(1).replace(".", "")
        try:
            harga = int(harga_str)
            query["harga_angka"] = (
                query["harga"]
                .astype(str)
                .str.replace("rp", "", case=False)
                .str.replace(".", "")
                .str.replace(",", "")
                .str.strip()
                .replace("", "0")
                .astype(float)
            )
            query = query[query["harga_angka"] <= harga]
        except:
            pass

    # Gabungkan nama mobil dan tahun untuk format ground truth
    ground_truth = query.apply(lambda row: f"{row['nama mobil']} ({row['tahun']})", axis=1).tolist()
    return ";".join(sorted(set(ground_truth)))

# 🔁 Buat ulang kolom ground_truth berdasarkan dataset
print("🔄 Menghasilkan ground truth untuk setiap pertanyaan...")
eval_df["ground_truth"] = eval_df["pertanyaan"].apply(filter_mobil)

# 💾 Simpan ulang CSV dengan kutip otomatis agar aman dari koma
eval_df.to_csv(
    "evaluasi_semua_batch.csv",
    index=False,
    quoting=csv.QUOTE_ALL  # ✅ Kunci penting agar data tetap utuh
)

print("✅ File evaluasi_semua_batch.csv berhasil diperbarui dengan kutip otomatis.")
