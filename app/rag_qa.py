# rag_qa.py (final fleksibel)
import json
import re
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

    # --- Ekstraksi filter ---
    filters = {}
    pertanyaan_lc = pertanyaan.lower()
    if "manual" in pertanyaan_lc:
        filters["transmisi"] = {"$eq": "manual"}
    if "matic" in pertanyaan_lc:
        filters["transmisi"] = {"$eq": "matic"}
    for bahan in ["diesel", "bensin", "hybrid", "listrik"]:
        if bahan in pertanyaan_lc:
            filters["bahan_bakar"] = {"$eq": bahan}

    match_usia = re.search(r"(di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan_lc)
    if match_usia:
        filters["usia"] = {"$lte": int(match_usia.group(2))}

    match_harga = re.search(r"(di ?bawah|maksimal|<=?) ?rp? ?(\d+[\.\d]*)", pertanyaan_lc)
    if match_harga:
        filters["harga_angka"] = {"$lte": int(match_harga.group(2).replace(".", ""))}

    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))

    try:
        filter_query = {"$and": [{k: v} for k, v in filters.items()]} if len(filters) > 1 else filters
        retriever = db.as_retriever(search_kwargs={"k": 30, "filter": filter_query})
        dokumen = await retriever.ainvoke(pertanyaan)
    except Exception:
        retriever = db.as_retriever(search_kwargs={"k": 10})
        dokumen = await retriever.ainvoke(pertanyaan)

    # --- Hilangkan duplikat berdasarkan nama mobil ---
    seen = set()
    unique_dokumen = []
    for doc in dokumen:
        nama = doc.page_content.split(",")[0].strip().split("(")[0].lower()
        if nama not in seen:
            unique_dokumen.append(doc)
            seen.add(nama)
    dokumen = unique_dokumen

    if not dokumen:
        fallback = "❌ Maaf, saya tidak menemukan mobil yang cocok. Silakan periksa ulang kriteria Anda."
        return fallback if not streaming else StreamingResponse(
            (f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n" for c in fallback),
            media_type="text/event-stream"
        )

    context = "\n".join(doc.page_content for doc in dokumen)
    prompt = PromptTemplate.from_template("""
Berikut adalah data mobil bekas yang tersedia dari database. Jawablah HANYA dari data berikut. Tidak boleh menambah, mengubah, atau mengarang informasi apapun yang tidak tercantum.

{context}

Instruksi:
- HANYA tampilkan mobil dari daftar di atas.
- TAMPILKAN SEMUA mobil yang memenuhi kriteria (jika banyak, tampilkan minimal 10).
- Tidak boleh mengarang merk/model/tahun/harga.
- Jika semua data tidak ada yang benar-benar sesuai, pilih yang paling mendekati, dan sebutkan alasan.
- Akhiri jawaban dengan kalimat: "Silakan bertanya lagi jika ada kriteria atau kebutuhan lain."

Format:
1. Nama Mobil
- Tahun: .
- Harga: .
- Usia: .
- Bahan Bakar: .
- Transmisi: .
- Alasan: Mobil ini sesuai karena .

Pertanyaan: {pertanyaan}
Jawaban:
""")
    
    llm = OllamaLLM(model="mistral", system=instruksi, stream=streaming)
    chain = prompt | llm | StrOutputParser()

    if not streaming:
        return await chain.ainvoke({"context": context, "pertanyaan": pertanyaan})

    async def event_gen():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        async for t in chain.astream({"context": context, "pertanyaan": pertanyaan}):
            yield f"data: {json.dumps({'type': 'stream', 'token': t})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# Untuk endpoint non-streaming
from fastapi import APIRouter
router = APIRouter()

@router.get("/jawab")
async def jawab_mobil(pertanyaan: str):
    hasil = await stream_mobil(pertanyaan, streaming=False)
    return JSONResponse(content={"jawaban": hasil})

@router.get("/stream")
async def stream_mobil_stream(pertanyaan: str):
    return await stream_mobil(pertanyaan, streaming=True)

