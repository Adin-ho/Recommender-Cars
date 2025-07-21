from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Load vector database dari folder
vector_store = Chroma(persist_directory="chroma")

# Jumlah vektor (record/data mobil)
jumlah_vektor = vector_store._collection.count()
print(f"Jumlah record mobil dalam ChromaDB: {jumlah_vektor}")

# Cek dimensi vektor (misal dengan embedding model)
embeddings = OllamaEmbeddings(model="mistral")  # atau model yang kamu pakai
vektor_sample = embeddings.embed_query("Contoh mobil Avanza 2022 matic")
print(f"Dimensi setiap vektor: {len(vektor_sample)}")

# Info: untuk cek ukuran file, lihat folder 'chroma/' di file explorer atau terminal
