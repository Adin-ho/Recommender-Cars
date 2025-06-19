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

    # Filter merk
    merk_list = ["bmw", "toyota", "honda", "mitsubishi", "suzuki", "nissan", "daihatsu", "mercedes", "wuling", "hyundai", "mazda", "kijang", "innova", "volkswagen", "vw"]
    selected_merk = None
    for merk in merk_list:
        if merk in pertanyaan_lc:
            filters["nama mobil"] = {"$regex": merk}
            selected_merk = merk
            break

    # Filter usia
    match_usia = re.search(r"(di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan_lc)
    if match_usia:
        filters["usia"] = {"$lte": int(match_usia.group(2))}

    # Filter harga/budget
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

    # Build mobil list
    mobil_list = []
    for doc in dokumen:
        data = doc.page_content.split(",")
        if len(data) < 6: continue
        try:
            nama = data[0].strip()
            harga = int(re.sub(r"\D", "", data[1]))
            tahun = int(re.sub(r"\D", "", data[2]))
            mobil_list.append({
                "nama": nama,
                "harga": harga,
                "tahun": tahun,
                "doc": doc
            })
        except:
            continue

    mobil_list = sorted(mobil_list, key=lambda x: (-x["tahun"], x["harga"]))

    # FILTER tahun muda (< 6 tahun, tahun_sekarang-6)
    tahun_sekarang = datetime.now().year
    tahun_minimal = tahun_sekarang - 6
    mobil_list_realistis = [m for m in mobil_list if m["tahun"] >= tahun_minimal]

    if len(mobil_list_realistis) < 4:
        mobil_list_realistis = [m for m in mobil_list if m["tahun"] >= tahun_sekarang - 10]
    if len(mobil_list_realistis) < 4:
        mobil_list_realistis = mobil_list[:20]  # Ambil 20 termuda

    # Filter ulang merk/budget super ketat (tidak boleh lewat)
    if selected_merk or budget_harga:
        mobil_list2 = []
        for m in mobil_list_realistis:
            cocok_merk = (selected_merk in m["nama"].lower()) if selected_merk else True
            cocok_harga = (m["harga"] <= budget_harga) if budget_harga else True
            if cocok_merk and cocok_harga:
                mobil_list2.append(m)
        mobil_list_realistis = mobil_list2

    dokumen = [m["doc"] for m in mobil_list_realistis]

    # Jika tidak ada mobil yang cocok, balas langsung tanpa prompt ke LLM
    if not dokumen:
        fallback = "❌ Maaf, saya tidak menemukan mobil yang cocok. Silakan periksa ulang kriteria Anda."
        return fallback if not streaming else StreamingResponse(
            (f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n" for c in fallback),
            media_type="text/event-stream"
        )

    context = "\n".join(doc.page_content for doc in dokumen)

    prompt = PromptTemplate.from_template(f"""
Berikut adalah data mobil bekas yang tersedia dari database. Jawablah HANYA dari data berikut. Tidak boleh menambah, mengubah, atau mengarang informasi apapun yang tidak tercantum.

{{context}}

Instruksi:
- Jawab HANYA dari data di atas. Tidak boleh membuat asumsi harga/tahun/merk baru.
- Tampilkan mobil PALING RELEVAN dan WORTH IT untuk tahun {{tahun_sekarang}}, utamakan:
    - Tahun muda ({{tahun_sekarang}} hingga {tahun_minimal}, maksimal 6 tahun ke belakang).
    - Harga masuk akal untuk pasar tahun sekarang.
    - Spesifikasi sesuai kriteria user (hemat BBM, transmisi, mesin bandel, fitur modern).
- Urutkan dari yang paling layak beli (tahun termuda dan harga pantas).
- Jika semua mobil sudah tua, tampilkan mobil tahun tertua paling tinggi di data saja.
- Setelah list, tutup jawaban dengan:
"Jika Anda punya preferensi khusus (tahun, fitur, merk, atau budget tertentu), silakan tanyakan agar saya bisa merekomendasikan yang lebih sesuai."
Format:
1. Nama Mobil
- Tahun: .
- Harga: .
- Usia: .
- Bahan Bakar: .
- Transmisi: .
- Alasan: Mobil ini layak dipertimbangkan karena .

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
