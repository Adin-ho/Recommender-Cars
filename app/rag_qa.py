import json
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langdetect import detect
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

async def stream_mobil(pertanyaan: str):
    bahasa = detect(pertanyaan)
    instruksi = {
        "id": "Jawab hanya dari data mobil berikut, jangan menebak. Jika tidak ditemukan, sarankan mobil dari data yang mendekati (misalnya lebih murah, lebih tua, atau lebih muda). Format jawaban dalam list rapi dengan alasan setiap rekomendasi.",
        "en": "Only answer based on provided car data. If not found, suggest cars with the closest criteria (e.g., cheaper, older, newer). Format in a clear list with reasoning.",
    }.get(bahasa, "Only use provided car data to answer.")

    # Load Chroma vector DB
    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))
    retriever = db.as_retriever()
    dokumen = await retriever.ainvoke(pertanyaan)
    context = "\n".join([doc.page_content for doc in dokumen])

    # Prompt Template
    template = ChatPromptTemplate.from_template(f"""
Anda adalah asisten rekomendasi mobil. Berikut adalah data mobil bekas:

{{context}}

Petunjuk:
- Pilih maksimal 7 mobil yang relevan dari data (jika tersedia).
- Jika tidak ada yang cocok, sarankan yang mendekati dari segi usia, harga, atau tahun produksi.
- Format:
  1. **[Nama Mobil]**
     - Tahun: ...
     - Harga: ...
     - Usia: ...
     - Bahan Bakar: ...
     - Transmisi: ...
     - Alasan: ...

Pertanyaan pengguna: {{pertanyaan}}

{instruksi}
""")

    chain = template | OllamaLLM(model="mistral") | StrOutputParser()
    output = await chain.ainvoke({
        "context": context,
        "pertanyaan": pertanyaan
    })

    # Streaming
    async def event_generator():
        yield f"data: {{\"type\": \"start\"}}\n\n"
        for char in output:
            yield f"data: {{\"type\": \"stream\", \"token\": {json.dumps(char)} }}\n\n"
        yield f"data: {{\"type\": \"end\"}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
