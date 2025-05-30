from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

# Ganti path dan nama collection sesuai setup kamu
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "mobil"

embedding = OllamaEmbeddings(model="mistral")
retriever = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding
).as_retriever()
