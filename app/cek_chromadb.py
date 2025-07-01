import chromadb
from chromadb.config import Settings

# Path ke folder database ChromaDB (default: './app/chroma_db')
CHROMA_DB_PATH = "./chroma"

def main():
    # Inisialisasi client persistent
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # List semua koleksi
    collections = client.list_collections()
    if not collections:
        print("Tidak ada koleksi ditemukan di database ChromaDB.")
        return

    print(f"Daftar koleksi di database:")
    for idx, col in enumerate(collections):
        print(f"{idx+1}. {col}")  # col sekarang adalah string nama koleksi


    # Pakai koleksi pertama (atau pilih sesuai kebutuhan)
    collection_name = collections[0]
    print(f"\nMengambil data dari koleksi: {collection_name}")

    collection = client.get_collection(collection_name)
    
    # Ambil maksimal 10 data
    results = collection.get(limit=10)

    print(f"\nMenampilkan maksimal 10 data pertama:")
    for i in range(len(results['ids'])):
        print("="*30)
        print(f"ID: {results['ids'][i]}")
        print(f"Document: {results['documents'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
        # Kalau ingin lihat embedding (vektor), uncomment baris berikut:
        # print(f"Embedding: {results['embeddings'][i]}")
    print("="*30)
    print(f"\nTotal data di koleksi ini (approx): {len(results['ids'])}")

if __name__ == "__main__":
    main()
