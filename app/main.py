from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import re
import requests
import asyncio

# ⬇️ Tambahkan ini SETELAH import FastAPI dan SEBELUM include_router
from app.rag_qa import router as rag_qa_router

app = FastAPI()  # ⬅️ Buat objek app di sini dulu!

# ⬇️ Sekarang baru include router LLM
app.include_router(rag_qa_router)

data_mobil = pd.read_csv('app/data/data_mobil.csv')
data_mobil.columns = data_mobil.columns.str.strip().str.lower()

if "harga_angka" not in data_mobil.columns:
    def bersihkan_harga(h):
        if pd.isna(h): return 0
        return int(re.sub(r'\D', '', str(h))) if re.search(r'\d', str(h)) else 0
    data_mobil["harga_angka"] = data_mobil["harga"].apply(bersihkan_harga)

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

@app.get("/stream")
async def stream(pertanyaan: str):
    jawaban = jawab(pertanyaan)
    async def event_stream():
        for word in jawaban.split():
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.06)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/jawab", response_class=PlainTextResponse)
def jawab(pertanyaan: str):
    hasil = data_mobil.copy()
    q = pertanyaan.lower()
    tahun_sekarang = 2025

    # 🎯 Filter usia
    match_usia = re.search(r"usia (?:di bawah|kurang dari) (\d+) tahun", q)
    if match_usia:
        batas_usia = int(match_usia.group(1))
        batas_tahun = tahun_sekarang - batas_usia
        hasil = hasil[hasil["tahun"] >= batas_tahun]

    # 🎯 Transmisi
    if "matic" in q:
        hasil = hasil[hasil["transmisi"].str.contains("matic", case=False, na=False)]
    if "manual" in q:
        hasil = hasil[hasil["transmisi"].str.contains("manual", case=False, na=False)]

    # 🎯 Bahan bakar
    for bb in ["diesel", "bensin", "hybrid", "listrik"]:
        if bb in q:
            hasil = hasil[hasil["bahan bakar"].str.contains(bb, case=False, na=False)]

    # 🎯 Harga
    match_harga = re.search(r"(?:di bawah|max(?:imal)?|<=?) ?rp? ?(\d[\d\.]*)", q)
    if match_harga:
        batas = int(match_harga.group(1).replace(".", ""))
        hasil = hasil[hasil["harga_angka"] <= batas]

    # 🎯 Tahun ke atas
    match_tahun_atas = re.search(r"tahun (\d{4}) ke atas", q)
    if match_tahun_atas:
        batas = int(match_tahun_atas.group(1))
        hasil = hasil[hasil["tahun"] >= batas]

    # 🎯 Tahun di bawah
    match_tahun_bawah = re.search(r"tahun (?:di bawah|kurang dari) (\d{4})", q)
    if match_tahun_bawah:
        batas = int(match_tahun_bawah.group(1))
        hasil = hasil[hasil["tahun"] < batas]

    # 🎯 Sinonim
    if "irit" in q or "hemat" in q:
        hasil = hasil[hasil["bahan bakar"].str.contains("bensin|hybrid", case=False, na=False)]

    # ❗️ Fallback ke LLM
    if hasil.empty:
        try:
            res = requests.get("http://localhost:8000/jawab_llm", params={"pertanyaan": pertanyaan})
            return res.json().get("jawaban", "tidak ditemukan")
        except:
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
    return unique_cars(output)
