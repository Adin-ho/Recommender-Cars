import json
import re
import pandas as pd
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect
from fastapi import APIRouter

router = APIRouter()

# === LOAD DATA SEKALI SAJA ===
df_mobil = pd.read_csv('app/data/data_mobil.csv', encoding="utf-8-sig")
df_mobil.columns = [c.strip().lower() for c in df_mobil.columns]

def clean_number(x):
    if pd.isna(x): return 0
    s = re.sub(r"\D", "", str(x))
    if not s: return 0
    return int(s)

for col in ["usia", "tahun"]:
    if col in df_mobil.columns:
        df_mobil[col] = df_mobil[col].apply(clean_number)
df_mobil["harga_angka"] = df_mobil["harga"].apply(clean_number)

async def stream_mobil(pertanyaan: str, streaming=True):
    bahasa = detect(pertanyaan)
    instruksi = {
        "id": "Jawab hanya berdasarkan data mobil berikut. Jangan buat asumsi atau menyebut info yang tidak ada. Jika tidak cocok, beri alternatif dari data.",
        "en": "Answer strictly based on the data below. Do not fabricate or assume unknown values. If there's no exact match, suggest alternatives from data.",
    }.get(bahasa, "Answer using the same language as the question and only based on the data below.")

    filters = {}
    pertanyaan_lc = pertanyaan.lower()
    if "manual" in pertanyaan_lc:
        filters["transmisi"] = "manual"
    if "matic" in pertanyaan_lc:
        filters["transmisi"] = "matic"
    for bahan in ["diesel", "bensin", "hybrid", "listrik"]:
        if bahan in pertanyaan_lc:
            filters["bahan bakar"] = bahan

    merk_list = [
        "bmw", "toyota", "honda", "mitsubishi", "suzuki", "nissan", "daihatsu",
        "mercedes", "wuling", "hyundai", "mazda", "kijang", "innova",
        "volkswagen", "vw", "alphard"
    ]
    selected_merk = None
    for merk in merk_list:
        if merk in pertanyaan_lc:
            filters["nama mobil"] = merk
            selected_merk = merk
            break

    match_usia = re.search(r"(di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan_lc)
    if match_usia:
        filters["usia"] = int(match_usia.group(2))

    # ===== DETEKSI HARGA =====
    match_harga = re.search(r"(\d+)[\s\.,]*juta", pertanyaan_lc)
    budget_harga = None
    harga_rentang = None
    if match_harga:
        budget_harga = int(match_harga.group(1).replace(".", "")) * 1_000_000
        filters["harga"] = budget_harga
        min_harga = int(budget_harga * 0.90)
        max_harga = int(budget_harga * 1.10)
        harga_rentang = (min_harga, max_harga)

    # ===== FILTER DATAFRAME =====
    df = df_mobil.copy()
    if "transmisi" in filters:
        df = df[df['transmisi'].str.lower() == filters["transmisi"]]
    if "bahan bakar" in filters:
        df = df[df['bahan bakar'].str.lower() == filters["bahan bakar"]]
    if "nama mobil" in filters:
        df = df[df['nama mobil'].str.lower().str.contains(filters["nama mobil"])]
    if "usia" in filters:
        df = df[df['usia'] <= filters["usia"]]

    df = df[df["nama mobil"].notnull() & (df["nama mobil"].str.strip() != "")]
    df = df.sort_values(['usia', 'tahun', 'harga_angka'], ascending=[True, False, True])

    info_alt = ""
    # ==== HANDLING ====
    if "nama mobil" in filters:
        # Tampilkan semua, max 10
        df_final = df.head(10)
        if df_final.empty:
            info_alt = "❌ Tidak ditemukan mobil sesuai kriteria nama mobil di database.\n\n"
    elif harga_rentang:
        df["selisih_harga"] = abs(df["harga_angka"] - budget_harga)
        df_budget = df[(df["harga_angka"] >= harga_rentang[0]) & (df["harga_angka"] <= harga_rentang[1])]
        df_budget = df_budget.sort_values("selisih_harga").head(5)
        if df_budget.empty:
            info_alt = f"Tidak ditemukan mobil di kisaran {budget_harga:,} (±10%). Berikut alternatif terdekat:\n"
            df_budget = df.sort_values("selisih_harga").head(5)
        df_final = df_budget
    else:
        mobil_muda = df[df['usia'] <= 5]
        if len(mobil_muda) >= 1:
            df_final = mobil_muda.head(5)
        else:
            df_final = df.head(5)
            info_alt = "PERHATIAN: Tidak ditemukan mobil usia muda (≤ 5 tahun) di database. Berikut alternatif tahun tua yang cocok untuk Anda:\n\n" if not df_final.empty else ""

    # === BUILD CONTEXT PASTI ADA NAMA MOBIL ===
    context = ""
    for idx, row in enumerate(df_final.to_dict(orient="records"), 1):
        context += (
            f"{idx}. Nama Mobil: {row.get('nama mobil','-')}\n"
            f"   - Tahun: {row.get('tahun','-')}\n"
            f"   - Harga: {row.get('harga','-')}\n"
            f"   - Usia: {row.get('usia','-')} tahun\n"
            f"   - Bahan Bakar: {row.get('bahan bakar','-')}\n"
            f"   - Transmisi: {row.get('transmisi','-')}\n"
            f"   - Kapasitas Mesin: {row.get('kapasitas mesin','-')}\n"
        )

    if not context.strip():
        fallback = "❌ Maaf, saya tidak menemukan mobil yang cocok. Silakan periksa ulang kriteria Anda."
        return fallback if not streaming else StreamingResponse(
            (f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n" for c in fallback),
            media_type="text/event-stream"
        )

    # ===== PROMPT SUPER STRICT =====
    prompt = PromptTemplate.from_template(f"""
{info_alt}Berikut adalah data mobil bekas yang tersedia dari database. Jawablah HANYA dari data berikut, **TIDAK BOLEH menambah, menghitung, atau mengubah nilai apapun**.

{{context}}

Instruksi penting:
- Semua field (Nama, Tahun, Harga, Usia, Bahan Bakar, Transmisi, Kapasitas Mesin) HARUS DIKUTIP APA ADANYA dari context, tanpa dihitung atau diubah!
- Field **Usia** WAJIB diambil dari baris 'Usia' di atas (tidak boleh dihitung ulang).
- Jika Nama mobil sudah ada tahun, tetap ambil field Tahun dari baris 'Tahun', bukan dari Nama.
- Urutkan jawaban sesuai urutan nomor pada context.
- Jika semua mobil sudah tua, tetap tampilkan maksimal 5 mobil yang paling muda.
- Setelah list, tutup jawaban dengan:
"Jika Anda punya preferensi khusus (tahun, fitur, merk, atau budget tertentu), silakan tanyakan agar saya bisa merekomendasikan yang lebih sesuai."

Format jawaban WAJIB seperti ini:
1. Nama Mobil: <Nama Mobil>
   - Tahun: <Tahun>
   - Harga: <Harga>
   - Usia: <Usia>
   - Bahan Bakar: <Bahan Bakar>
   - Transmisi: <Transmisi>
   - Alasan: Mobil ini layak dipertimbangkan karena ... (isi bebas, tidak boleh ngarang field!)

Pertanyaan: {{pertanyaan}}
Jawaban:
""")

    llm = OllamaLLM(model="mistral", system=instruksi, stream=streaming)
    chain = prompt | llm | StrOutputParser()

    if not streaming:
        return await chain.ainvoke({
            "context": context,
            "pertanyaan": pertanyaan,
        })

    async def event_gen():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        async for t in chain.astream({
            "context": context,
            "pertanyaan": pertanyaan,
        }):
            yield f"data: {json.dumps({'type': 'stream', 'token': t})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@router.get("/jawab")
async def jawab_mobil(pertanyaan: str):
    hasil = await stream_mobil(pertanyaan, streaming=False)
    return JSONResponse(content={"jawaban": hasil})

@router.get("/stream")
async def stream_mobil_stream(pertanyaan: str):
    return await stream_mobil(pertanyaan, streaming=True)
