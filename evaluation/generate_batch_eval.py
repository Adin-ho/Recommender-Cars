import pandas as pd
import random
import csv

random.seed(42)

templates = [
    "mobil {transmisi} {bahan_bakar} dengan usia di bawah {usia} tahun",
    "mobil {bahan_bakar} tahun {tahun}+",
    "mobil dengan harga di bawah Rp {harga}",
    "mobil {transmisi} tahun {tahun}+ dengan harga di bawah Rp {harga}",
    "mobil {bahan_bakar} dan transmisi {transmisi} dengan harga maksimal Rp {harga}",
    "mobil tahun {tahun}+ yang menggunakan {bahan_bakar}",
    "mobil usia di bawah {usia} tahun dan harga di bawah Rp {harga}",
    "mobil {bahan_bakar} usia kurang dari {usia} tahun",
    "mobil {transmisi} dengan harga maksimal Rp {harga}",
    "mobil {bahan_bakar} tahun {tahun}+ dan usia di bawah {usia} tahun",
]

transmisi_opsi = ["manual", "matic"]
bahan_bakar_opsi = ["bensin", "diesel", "hybrid", "listrik"]
tahun_opsi = list(range(2015, 2025))
usia_opsi = list(range(1, 11))
harga_opsi = [100_000_000, 150_000_000, 200_000_000, 250_000_000, 300_000_000,
                350_000_000, 400_000_000, 500_000_000, 600_000_000]

pertanyaan = set()
while len(pertanyaan) < 500:
    template = random.choice(templates)
    t = template.format(
        transmisi=random.choice(transmisi_opsi),
        bahan_bakar=random.choice(bahan_bakar_opsi),
        tahun=random.choice(tahun_opsi),
        usia=random.choice(usia_opsi),
        harga=f"{random.choice(harga_opsi):,}".replace(",", ".")
    )
    pertanyaan.add(t.lower())

df = pd.DataFrame({"pertanyaan": list(pertanyaan)})
df.to_csv("evaluasi_semua_batch.csv", index=False, quoting=csv.QUOTE_ALL)
print("✅ evaluasi_semua_batch.csv berhasil dibuat dengan 500 pertanyaan.")
