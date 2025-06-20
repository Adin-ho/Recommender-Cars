import json
import re
from datetime import datetime
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect

async def stream_mobil(pertanyaan: str, streaming=True):
    bahasa = detect(pertanyaan)
    instruksi = {
        "id": "Jawab hanya berdasarkan data mobil berikut. Jangan buat asumsi atau menyebut info yang tidak ada. Jika tidak cocok, beri alternatif dari data.",
        "en": "Answer strictly based on the data below. Do not fabricate or assume unknown values. If there's no exact match, suggest alternatives from data.",
    }.get(bahasa, "Answer using the same language as the question and only based on the data below.")

    filters = {}
    pertanyaan_lc = pertanyaan.lower()
    if "manual" in pertanyaan_lc:
        filters["transmisi"] = {"$eq": "manual"}
    if "matic" in pertanyaan_lc:
        filters["transmisi"] = {"$eq": "matic"}
    for bahan in ["diesel", "bensin", "hybrid", "listrik"]:
        if bahan in pertanyaan_lc:
            filters["bahan_bakar"] = {"$eq": bahan}

    merk_list = ["bmw", "toyota", "honda", "mitsubishi", "suzuki", "nissan", "daihatsu", "mercedes", "wuling", "hyundai", "mazda", "kijang", "innova", "volkswagen", "vw"]
    selected_merk = None
    for merk in merk_list:
        if merk in pertanyaan_lc:
            filters["nama mobil"] = {"$regex": merk}
            selected_merk = merk
            break

    match_usia = re.search(r"(di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan_lc)
    if match_usia:
        filters["usia"] = {"$lte": int(match_usia.group(2))}

    match_harga = re.search(r"(di ?bawah|maksimal|budget|bujet|<=?) ?rp? ?(\d+[\.\d]*)", pertanyaan_lc)
    budget_harga = None
    if match_harga:
        budget_harga = int(match_harga.group(2).replace(".", ""))
        filters["harga_angka"] = {"$lte": budget_harga}

    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))
    try:
        filter_query = {"$and": [{k: v} for k, v in filters.items()]} if len(filters) > 1 else filters
        retriever = db.as_retriever(search_kwargs={"k": 120, "filter": filter_query})
        dokumen = await retriever.ainvoke(pertanyaan)
    except Exception:
        retriever = db.as_retriever(search_kwargs={"k": 80})
        dokumen = await retriever.ainvoke(pertanyaan)

    # Dedup hanya nama+TAHUN agar semua varian unik bisa masuk
    seen = set()
    unique_dokumen = []
    for doc in dokumen:
        try:
            data = doc.page_content.split(",")
            nama = data[0].strip().lower()
            tahun = re.sub(r"\D", "", data[2])
            key = f"{nama}|{tahun}"
        except:
            key = doc.page_content.strip().lower()
        if key not in seen:
            unique_dokumen.append(doc)
            seen.add(key)
    dokumen = unique_dokumen

    # Build mobil list + HITUNG USIA!
    tahun_sekarang = datetime.now().year
    mobil_list = []
    for doc in dokumen:
        data = doc.page_content.split(",")
        if len(data) < 6: continue
        try:
            nama = data[0].strip()
            harga = data[1].strip()
            tahun = data[2].strip()
            bahan_bakar = data[3].strip()
            transmisi = data[4].strip()
            mesin = data[5].strip()
            usia = ""
            if tahun.isdigit():
                usia = str(max(0, tahun_sekarang - int(tahun)))
            mobil_list.append({
                "nama": nama,
                "harga": harga,
                "tahun": tahun,
                "bahan_bakar": bahan_bakar,
                "transmisi": transmisi,
                "mesin": mesin,
                "usia": usia,
                "doc": doc
            })
        except:
            continue

    # Sort: Tahun termuda -> harga termurah
    mobil_list = sorted(
        mobil_list,
        key=lambda x: (
            -int(x["tahun"]) if x["tahun"].isdigit() else 0,
            int(re.sub(r"\D", "", x["harga"] or "0"))
        )
    )

    # Filter tahun minimal (misal 6 tahun terakhir), lalu LIMIT ke 5 saja!
    tahun_minimal = tahun_sekarang - 6
    mobil_list_realistis = [m for m in mobil_list if m["tahun"].isdigit() and int(m["tahun"]) >= tahun_minimal]
    if len(mobil_list_realistis) < 5:
        mobil_list_realistis = [m for m in mobil_list if m["tahun"].isdigit() and int(m["tahun"]) >= tahun_sekarang - 10]
    if len(mobil_list_realistis) < 5:
        mobil_list_realistis = mobil_list[:10]
    # LIMIT ke 5 saja!
    mobil_list_realistis = mobil_list_realistis[:5]

    # Filter ulang merk/budget super ketat (tidak boleh lewat)
    if selected_merk or budget_harga:
        mobil_list2 = []
        for m in mobil_list_realistis:
            cocok_merk = (selected_merk in m["nama"].lower()) if selected_merk else True
            harga_angka = int(re.sub(r"\D", "", m["harga"] or "0")) if m["harga"] else 0
            cocok_harga = (harga_angka <= budget_harga) if budget_harga else True
            if cocok_merk and cocok_harga:
                mobil_list2.append(m)
        mobil_list_realistis = mobil_list2[:5]  # Pastikan tetap maksimal 5

    # Context per field, SUDAH termuat usia!
    context = "\n".join(
        f"Nama: {m['nama']} | Harga: {m['harga']} | Tahun: {m['tahun']} | Usia: {m['usia']} tahun | Bahan Bakar: {m['bahan_bakar']} | Transmisi: {m['transmisi']} | Mesin: {m['mesin']}"
        for m in mobil_list_realistis
    )

    # Fallback jika context kosong
    if not context.strip():
        fallback = "❌ Maaf, saya tidak menemukan mobil yang cocok. Silakan periksa ulang kriteria Anda."
        return fallback if not streaming else StreamingResponse(
            (f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n" for c in fallback),
            media_type="text/event-stream"
        )

    prompt = PromptTemplate.from_template(f"""
Berikut adalah data mobil bekas yang tersedia dari database. Jawablah HANYA dari data berikut. Tidak boleh menambah, mengubah, atau mengarang informasi apapun yang tidak tercantum.

{{context}}

Instruksi penting:
- Semua jawaban HARUS mengutip field yang tersedia di atas secara apa adanya, tidak boleh memparafrase/mengganti.
- Tahun, harga, dan spesifikasi mobil WAJIB diambil dari field context (misal: Tahun: ...), **BUKAN dari nama atau instruksi atau tahun sekarang**.
- Usia mobil WAJIB diambil dari field Usia di atas (bukan dihitung olehmu!).
- Jika Nama mobil sudah ada tahun (misal: “BMW 320i At (2022)”), field Tahun TETAP diambil dari field Tahun, bukan dari nama.
- Urutkan mobil dari tahun termuda dan harga yang paling pantas.
- Jika semua mobil sudah tua, tetap tampilkan maksimal 5 mobil yang paling muda.
- Setelah list, tutup jawaban dengan:
"Jika Anda punya preferensi khusus (tahun, fitur, merk, atau budget tertentu), silakan tanyakan agar saya bisa merekomendasikan yang lebih sesuai."

Format jawaban WAJIB seperti ini:
1. Nama Mobil
- Tahun: <Tahun dari field Tahun>
- Harga: <Harga dari field Harga>
- Usia: <dari field Usia>
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
            "tahun_sekarang": tahun_sekarang
        })

    async def event_gen():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        async for t in chain.astream({
            "context": context,
            "pertanyaan": pertanyaan,
            "tahun_sekarang": tahun_sekarang
        }):
            yield f"data: {json.dumps({'type': 'stream', 'token': t})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

from fastapi import APIRouter
router = APIRouter()

@router.get("/jawab")
async def jawab_mobil(pertanyaan: str):
    hasil = await stream_mobil(pertanyaan, streaming=False)
    return JSONResponse(content={"jawaban": hasil})

@router.get("/stream")
async def stream_mobil_stream(pertanyaan: str):
    return await stream_mobil(pertanyaan, streaming=True)
