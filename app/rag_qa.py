import json
import re
import pandas as pd
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect
from fastapi import APIRouter, Query

router = APIRouter()

# === LOAD DATA SEKALI SAJA ===
df_mobil = pd.read_csv('app/data/data_mobil_final.csv', encoding="utf-8-sig")
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

def extract_exclude_list(pertanyaan):
    pertanyaan = pertanyaan.lower()
    exclude = []
    patterns = [
        r"(?:selain|kecuali|bukan)\s*([^\?\.]+)"
    ]
    for pat in patterns:
        match = re.search(pat, pertanyaan)
        if match:
            kandidat = re.split(r',| dan | atau ', match.group(1))
            for k in kandidat:
                nama = k.strip()
                if nama and len(nama) > 2:
                    exclude.append(nama)
    return [x for x in set([e.strip() for e in exclude if e])]

async def stream_mobil(pertanyaan: str, streaming=True, exclude_param: str = ""):
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

    # Ambil dari pertanyaan (selain/kecuali) + ambil dari parameter exclude
    exclude_list = extract_exclude_list(pertanyaan_lc)
    if exclude_param:
        exclude_list += [x.strip().lower() for x in exclude_param.split(",") if x.strip()]

    exclude_list = list(set([e for e in exclude_list if e]))  # pastikan unik

    df = df_mobil.copy()
    try:
        if "transmisi" in filters:
            df = df[df['transmisi'].str.lower() == filters["transmisi"]]
        if "bahan bakar" in filters:
            df = df[df['bahan bakar'].str.lower() == filters["bahan bakar"]]
        if "nama mobil" in filters:
            df = df[df['nama mobil'].str.lower().str.contains(filters["nama mobil"])]
        if "usia" in filters:
            df = df[df['usia'] <= filters["usia"]]
    except Exception:
        pass

    # Exclude logic: filter semua nama mobil yang sudah ada di exclude_list
    for excl in exclude_list:
        df = df[~df['nama mobil'].str.lower().str.contains(excl)]

    df = df[df["nama mobil"].notnull() & (df["nama mobil"].str.strip() != "")]
    df = df.sort_values(['tahun', 'usia', 'harga_angka'], ascending=[False, True, True])

    info_alt = ""
    if harga_rentang:
        df_final = df[(df["harga_angka"] >= harga_rentang[0]) & (df["harga_angka"] <= harga_rentang[1])]
        if len(df_final) < 5:
            df_bawah = df[df["harga_angka"] < harga_rentang[0]].sort_values("harga_angka", ascending=False)
            df_final = pd.concat([df_final, df_bawah.head(5 - len(df_final))])
        if len(df_final) < 5:
            df_atas = df[df["harga_angka"] > harga_rentang[1]].sort_values("harga_angka", ascending=True)
            df_final = pd.concat([df_final, df_atas.head(5 - len(df_final))])
        df_final = df_final.head(5)
        if df_final.empty:
            info_alt = f"Tidak ditemukan mobil di kisaran {budget_harga:,} (±10%). Berikut alternatif terdekat:\n"
            df_final = df.sort_values("harga_angka").head(5)
    else:
        tahun_query = None
        match_tahun = re.search(r"(tahun|th|thn)[\s:]*([0-9]{4})", pertanyaan_lc)
        if match_tahun:
            tahun_query = int(match_tahun.group(2))
        else:
            for thn in range(2024, 2031):
                if str(thn) in pertanyaan_lc:
                    tahun_query = thn
                    break

        df_final = pd.DataFrame()
        if tahun_query:
            df_utama = df[df['tahun'] == tahun_query]
            if not df_utama.empty:
                df_final = df_utama
        if df_final.empty or len(df_final) < 5:
            df_5th = df[df['usia'] <= 5]
            if not df_5th.empty:
                df_5th = df_5th[~df_5th.index.isin(df_final.index)]
                df_final = pd.concat([df_final, df_5th])
        if len(df_final) < 5:
            df_lain = df[~df.index.isin(df_final.index)]
            df_final = pd.concat([df_final, df_lain.head(5 - len(df_final))])
        df_final = df_final.head(5)

    if df_final.empty:
        exclude_info = ""
        if exclude_list:
            exclude_info = f"selain {', '.join(exclude_list)}"
        fallback = f"❌ Maaf, {('' if not exclude_info else exclude_info+', ')}tidak ada mobil lain yang sesuai di database."
        return fallback if not streaming else StreamingResponse(
            (f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n" for c in fallback),
            media_type="text/event-stream"
        )

    def safe_get(row, key):
        value = row.get(key, '-')
        if pd.isna(value) or str(value).strip() == '':
            return '-'
        return value

    context = ""
    for idx, row in enumerate(df_final.to_dict(orient="records"), 1):
        context += (
            f"{idx}. Nama Mobil: {safe_get(row, 'nama mobil')}\n"
            f"   - Tahun: {safe_get(row, 'tahun')}\n"
            f"   - Harga: {safe_get(row, 'harga')}\n"
            f"   - Usia: {safe_get(row, 'usia')} tahun\n"
            f"   - Bahan Bakar: {safe_get(row, 'bahan bakar')}\n"
            f"   - Transmisi: {safe_get(row, 'transmisi')}\n"
            f"   - Kapasitas Mesin: {safe_get(row, 'kapasitas mesin')}\n"
        )

    if not context.strip():
        fallback = "❌ Maaf, saya tidak menemukan mobil yang cocok. Silakan periksa ulang kriteria Anda."
        return fallback if not streaming else StreamingResponse(
            (f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n" for c in fallback),
            media_type="text/event-stream"
        )

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
async def jawab_mobil(pertanyaan: str, exclude: str = Query("", description="Nama mobil yang sudah direkomendasikan, dipisahkan koma.")):
    hasil = await stream_mobil(pertanyaan, streaming=False, exclude_param=exclude)
    return JSONResponse(content={"jawaban": hasil})

@router.get("/stream")
async def stream_mobil_stream(pertanyaan: str, exclude: str = Query("", description="Nama mobil yang sudah direkomendasikan, dipisahkan koma.")):
    return await stream_mobil(pertanyaan, streaming=True, exclude_param=exclude)
