import chromadb

CHROMA_DB_PATH = "./chroma"

def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = client.list_collections()
    if not collections:
        print("Tidak ada koleksi ditemukan di database ChromaDB.")
        return

    print(f"Daftar koleksi di database:")
    for idx, col in enumerate(collections):
        print(f"{idx+1}. {col}")

    collection_name = collections[0]  # Sekarang col adalah string nama koleksi
    print(f"\nMengambil data dari koleksi: {collection_name}")

    collection = client.get_collection(collection_name)
    results = collection.get(limit=10)

    print(f"\nMenampilkan maksimal 10 data pertama:")
    for i in range(len(results['ids'])):
        print("="*30)
        print(f"ID: {results['ids'][i]}")
        print(f"Document: {results['documents'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
    print("="*30)
    print(f"\nTotal data di koleksi ini (approx): {len(results['ids'])}")

if __name__ == "__main__":
    main()
