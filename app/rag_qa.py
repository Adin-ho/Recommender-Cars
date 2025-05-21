import json
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

    # Deteksi filter bahan bakar otomatis dari pertanyaan
    bahan_bakar_filter = None
    for keyword in ["hybrid", "bensin", "diesel", "listrik"]:
        if keyword in pertanyaan.lower():
            bahan_bakar_filter = keyword.lower()
            break

    # Setup Chroma dan search_kwargs
    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))
    retriever = None

    if bahan_bakar_filter:
        retriever = db.as_retriever(search_kwargs={
            "filter": {"bahan_bakar": {"$eq": bahan_bakar_filter}}
        })
    else:
        retriever = db.as_retriever()

    dokumen = await retriever.ainvoke(pertanyaan)

    # Jika tidak ditemukan dokumen
    if not dokumen:
        async def fallback_gen():
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            msg = (
                f"❌ Maaf, tidak ditemukan mobil dengan bahan bakar {bahan_bakar_filter} dalam data. "
                "Ingin mencari berdasarkan harga, usia, atau jenis transmisi?"
                if bahan_bakar_filter else
                "❌ Maaf, tidak ditemukan mobil yang sesuai dari data. Silakan berikan kriteria tambahan."
            )
            for c in msg:
                yield f"data: {json.dumps({'type': 'stream', 'token': c})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        return StreamingResponse(fallback_gen(), media_type="text/event-stream")

    context = "\n".join([doc.page_content for doc in dokumen])

    # Prompt yang ketat, akurat, dan adaptif
    prompt = PromptTemplate.from_template("""
    Berikut adalah data mobil bekas yang tersedia. Jangan menyebut mobil atau data yang tidak ada di daftar ini. Jangan membuat saran tambahan jika tidak ditemukan dalam daftar.

    {context}

    Instruksi:
    - Pilih mobil rekomendasi yang paling sesuai dengan pertanyaan pengguna tidak ada batas maksimal.
    - Jika tidak ada mobil yang benar-benar memenuhi, cukup jawab dalam bentuk paragraf, misalnya:
    - Menanyakan ulang kriteria pengguna
    - Bertanya untuk keperluan penggunaan mobil (misal: keluarga, kerja, off-road)
    - Atau menyarankan pengguna mempertimbangkan kriteria lain (harga, tahun, dll)
    - Jangan menyebut atau menulis nama mobil yang tidak disebut dalam data di atas.
    - Format list rapi hanya jika ada mobil valid. Jika tidak, gunakan format paragraf.

    Pertanyaan pengguna: {pertanyaan}
    Jawaban:
    """)


    llm = OllamaLLM(model="mistral", system=instruksi, stream=True)
    chain = prompt | llm | StrOutputParser()

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        async for token in chain.astream({
            "context": context,
            "pertanyaan": pertanyaan
        }):
            yield f"data: {json.dumps({'type': 'stream', 'token': token})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
