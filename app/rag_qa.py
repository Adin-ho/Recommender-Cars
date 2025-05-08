# app/rag_qa.py
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from fastapi.responses import StreamingResponse
from langchain.chains import RetrievalQA
from langdetect import detect
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def tanya_mobil(pertanyaan: str) -> str:
    # Load database dan retriever
    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))
    retriever = db.as_retriever()

    # Ambil dokumen relevan
    dokumen = retriever.get_relevant_documents(pertanyaan)
    context = "\n".join([doc.page_content for doc in dokumen])

    # Buat prompt manual
    prompt = f"""
Saya sedang mencari rekomendasi mobil.
Pertanyaan saya: {pertanyaan}

Berikut data mobil yang mungkin relevan:
{context}

Tolong beri rekomendasi mobil terbaik berdasarkan data di atas.
"""

    llm = OllamaLLM(model="mistral")
    jawaban = llm.invoke(prompt)
    return jawaban

async def stream_mobil(pertanyaan: str):
    bahasa = detect(pertanyaan)
    instruksi = {
        "id": "Jawab dalam bahasa Indonesia.",
        "en": "Please answer in English.",
    }.get(bahasa, "Please answer in the same language as the question.")

    # Persiapkan data
    db = Chroma(persist_directory="chroma", embedding_function=OllamaEmbeddings(model="mistral"))
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(pertanyaan)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate.from_template("""
{instruksi}

Pertanyaan saya: {pertanyaan}

Berikut data mobil yang mungkin relevan:
{context}

Tolong beri rekomendasi mobil terbaik berdasarkan data di atas.
""")

    # Rangkaian LLM yang mendukung streaming
    llm = OllamaLLM(model="mistral", stream=True)
    chain = prompt | llm | StrOutputParser()

    # Jalankan streaming
    async for chunk in chain.astream({
        "instruksi": instruksi,
        "pertanyaan": pertanyaan,
        "context": context,
    }):
        yield chunk if isinstance(chunk, str) else str(chunk)
