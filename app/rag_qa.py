import json
from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langdetect import detect

async def stream_mobil(pertanyaan: str):
    # Deteksi bahasa
    bahasa = detect(pertanyaan)
    instruksi = {
        "id": "Jawab dalam bahasa Indonesia. Jangan menjawab jika data tidak ditemukan.",
        "en": "Answer in English. Do not fabricate if data is missing."
    }.get(bahasa, "Answer in the same language as the question.")

    # Inisialisasi retriever dan filter dokumen relevan
    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))
    retriever = db.as_retriever()
    dokumen = retriever.get_relevant_documents(pertanyaan)

    # ❗ Filter dokumen jika pertanyaan butuh konteks spesifik (contoh: hanya mobil diesel)
    if "diesel" in pertanyaan.lower():
        dokumen = [doc for doc in dokumen if "diesel" in doc.page_content.lower()]

    # Siapkan context
    if not dokumen:
        response_text = "Maaf, tidak ada data mobil yang cocok ditemukan."
    else:
        context = "\n".join([doc.page_content for doc in dokumen])

        prompt = f"""
Anda adalah asisten yang hanya memberikan jawaban berdasarkan data.
Berikut adalah data mobil bekas:
{context}

JAWAB HANYA BERDASARKAN DATA. Jika tidak ada data yang cocok, jawab: "Maaf, tidak ada data yang cocok."
Pertanyaan: {pertanyaan}
Jawaban:
"""
        llm = OllamaLLM(model="mistral", system=instruksi)
        response_text = llm.invoke(prompt)

    # Streaming SSE
    async def event_generator():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        for char in response_text:
            yield f"data: {json.dumps({'type': 'stream', 'token': char})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
