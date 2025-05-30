from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import re

app = FastAPI()
data_mobil = pd.read_csv('app/data/data_mobil.csv')
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

def unique_cars(output):
    found = re.findall(r"([a-z0-9 .\-]+)\s*\((\d{4})\)", output.lower())
    seen = set()
    cars = []
    for n, t in found:
        key = f"{n.strip()} ({t})"
        if key not in seen:
            seen.add(key)
            cars.append(key)
    return "; ".join(cars)

@app.get("/jawab", response_class=PlainTextResponse)
def jawab(pertanyaan: str):
    hasil = data_mobil.copy()
    q = pertanyaan.lower()

    # Filter transmisi dan bahan bakar
    if "matic" in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
    if "manual" in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]
    for bb in ["diesel", "bensin", "hybrid", "listrik"]:
        if bb in q:
            hasil = hasil[hasil["bahan bakar"].str.contains(bb, case=False, na=False)]

    # Filter harga
    match_harga = re.search(r"(?:di bawah|maximal|<=?) ?rp? ?(\d+[\.\d]*)", q)
    if match_harga:
        batas = int(match_harga.group(1).replace(".", ""))
        if "harga_angka" in hasil.columns:
            hasil = hasil[hasil["harga_angka"] <= batas]

    # Filter tahun mobil
    match_tahun = re.search(r"tahun (\d{4}) ke atas", q)
    if match_tahun:
        batas_tahun = int(match_tahun.group(1))
        hasil = hasil[hasil["tahun"] >= batas_tahun]

    if hasil.empty:
        return "tidak ditemukan"

    def bersih_nama(nama, tahun):
        nama = nama.strip().lower()
        tahun = str(tahun).strip()
        nama = re.sub(r"\s*\(\d{4}\)$", "", nama)
        return f"{nama} ({tahun})"

    output = "; ".join(
        bersih_nama(row['nama mobil'], row['tahun'])
        for _, row in hasil.head(5).iterrows()
    )
    output = unique_cars(output)

    return output
