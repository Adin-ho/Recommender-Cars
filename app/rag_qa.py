from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def buat_rag_chain():
    embeddings = OllamaEmbeddings(model="mistral")
    vectordb = Chroma(persist_directory="chroma", embedding_function=embeddings)
    llm = Ollama(model="mistral")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

def jawab_pertanyaan(pertanyaan: str):
    chain = buat_rag_chain()
    hasil = chain(pertanyaan)
    return {
        "jawaban": hasil["result"],
        "sumber": [doc.page_content for doc in hasil["source_documents"]]
    }
