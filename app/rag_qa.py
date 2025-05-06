# app/rag_qa.py
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

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
