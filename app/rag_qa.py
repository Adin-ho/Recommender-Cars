from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from fastapi.responses import StreamingResponse
from langdetect import detect
import json

async def stream_mobil(pertanyaan: str):
    bahasa = detect(pertanyaan)
    instruksi = {
        "id": "Jawab dalam bahasa Indonesia.",
        "en": "Please answer in English.",
    }.get(bahasa, "Please answer in the same language as the question.")

    # Ambil dokumen dari Chroma
    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))
    retriever = db.as_retriever()
    docs = retriever.invoke(pertanyaan)  # pakai invoke(), bukan get_relevant_documents()
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Anda adalah asisten rekomendasi mobil bekas berdasarkan data.
Berikut data mobil tersedia:
{context}

Jawab hanya berdasarkan data. Jangan buat asumsi jika tidak ada data cocok.
Pertanyaan pengguna: {pertanyaan}

Jawaban:
"""

    llm = OllamaLLM(model="mistral", system=instruksi)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        async for chunk in llm.astream(prompt):
            yield f"data: {json.dumps({'type': 'stream', 'token': chunk})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
