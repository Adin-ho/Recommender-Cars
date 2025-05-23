import json
import re
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langdetect import detect
from fastapi.responses import StreamingResponse
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

async def stream_mobil(pertanyaan: str):
    bahasa = detect(pertanyaan)
    instruksi = {
        "id": "Jawab hanya berdasarkan data mobil berikut. Jangan buat asumsi atau menyebut info yang tidak ada. Jika tidak cocok, beri alternatif dari data.",
        "en": "Answer strictly based on the data below. Do not fabricate or assume unknown values. If there's no exact match, suggest alternatives from data.",
    }.get(bahasa, "Answer using the same language as the question and only based on the data below.")

    # Parsing filter dari pertanyaan
    filters = {}
    pertanyaan_lc = pertanyaan.lower()
    if "manual" in pertanyaan_lc:
        filters["transmisi"] = {"$eq": "manual"}
    if "matic" in pertanyaan_lc:
        filters["transmisi"] = {"$eq": "matic"}
    for bahan in ["diesel", "bensin", "hybrid", "listrik"]:
        if bahan in pertanyaan_lc:
            filters["bahan_bakar"] = {"$eq": bahan}

    usia_match = re.search(r"(di ?bawah|kurang dari) (\d{1,2}) tahun", pertanyaan_lc)
    if usia_match:
        usia_max = int(usia_match.group(2))
        filters["usia"] = {"$lte": usia_max}

    harga_match = re.search(r"(di ?bawah|maksimal|<=?) ?rp? ?(\d+[\.\d]*)", pertanyaan_lc)
    if harga_match:
        harga = int(harga_match.group(2).replace(".", ""))
        filters["harga_angka"] = {"$lte": harga}

    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))

    # Perbaikan: gunakan $and agar Chroma tidak error
    if filters:
        filter_query = {"$and": [{k: v} for k, v in filters.items()]}
        retriever = db.as_retriever(search_kwargs={"k": 10, "filter": filter_query})
    else:
        retriever = db.as_retriever(search_kwargs={"k": 10})

    retriever = db.as_retriever(search_kwargs={"k": 10, "filter": filters} if filters else {"k": 10})

    dokumen = await retriever.ainvoke(pertanyaan)

    # Dedup berdasarkan nama mobil
    unique_dokumen = []
    seen_mobil = set()
    for doc in dokumen:
        first_line = doc.page_content.split(",")[0].strip()
        nama_mobil = first_line.split("(")[0].strip().lower()
        if nama_mobil not in seen_mobil:
            unique_dokumen.append(doc)
            seen_mobil.add(nama_mobil)
    dokumen = unique_dokumen

    if not dokumen:
        async def fallback_gen():
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            msg = "❌ Maaf, tidak ditemukan mobil yang cocok. Anda bisa menambahkan kriteria lain seperti harga, tahun, atau jenis transmisi."
            for c in msg:
                yield f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        return StreamingResponse(fallback_gen(), media_type="text/event-stream")

    context = "\n".join([doc.page_content for doc in dokumen])
    prompt = PromptTemplate.from_template("""
    Berikut adalah data mobil bekas yang tersedia dari database. Gunakan hanya informasi dari data berikut, dan jangan menyebut atau membuat mobil yang tidak disebutkan di dalam data.

    {context}

    Instruksi:
    - Tampilkan minimal 2 mobil. Tidak ada batas maksimal jumlah mobil jika datanya relevan dan sesuai kriteria pertanyaan pengguna.
    - Semua mobil yang ditampilkan HARUS memenuhi SELURUH kriteria eksplisit dari pengguna.
    - Jangan tampilkan mobil lebih dari satu kali.
    - Jangan buat item list tambahan jika tidak ada mobil lain.
    - Jika hanya sedikit mobil yang sesuai, tetap tampilkan dan beri alasan logis, contohnya:
    - Usia mobil di atas 6 tahun tetap dapat dipertimbangkan karena harga lebih terjangkau atau kualitasnya.
    - Jika tidak ada mobil yang cocok, tuliskan paragraf singkat menanyakan ulang kebutuhan pengguna atau ajukan pertanyaan tindak lanjut.
    - Setelah menampilkan list, tambahkan kalimat penjelas/elaborasi/transisi seperti gaya GPT.
    Contoh:
    "Jika Anda ingin mengeksplorasi pilihan lain dengan kriteria berbeda, saya bisa bantu mencarikan opsi yang sesuai."
    atau:
    "Tentu, saya siap bantu jika Anda ingin fokus pada aspek lain seperti merek, kapasitas mesin, atau fitur tambahan."
    - Jangan menulis kalimat promosi seperti "kami sedang memperluas database" atau "kami melayani seluruh Indonesia".

    Gunakan format list seperti:

    1. **[Nama Mobil]**
    - Tahun: ...
    - Harga: ...
    - Usia: ...
    - Bahan Bakar: ...
    - Transmisi: ...
    - Alasan: Mobil ini sesuai karena ...

    Pertanyaan pengguna: {pertanyaan}
    Jawaban:
    """)

    llm = OllamaLLM(model="mistral", system=instruksi, stream=True)
    chain = prompt | llm | StrOutputParser()

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        async for token in chain.astream({"context": context, "pertanyaan": pertanyaan}):
            yield f"data: {json.dumps({'type': 'stream', 'token': token})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")