import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm  # Tambahkan ini

def simpan_vektor_mobil():
    print("[INFO] Membaca dataset...")
    df = pd.read_csv("app/data/data_mobil.csv")

    print("[INFO] Mulai proses embedding...")
    dokumen = [
        f"{row['Nama Mobil']} ({row['Tahun']}), usia {row['Usia Kendaraan (tahun)']} tahun, "
        f"harga {row['Harga']}, transmisi {row['Transmisi']}, bahan bakar {row['Bahan Bakar']}, "
        f"kapasitas mesin {row['Kapasitas Mesin (cc)']} cc"
        for _, row in tqdm(df.iterrows(), total=len(df))  # pakai tqdm
    ]

    embeddings = OllamaEmbeddings(model="mistral")
    print("[INFO] Menyimpan ke ChromaDB...")
    Chroma.from_texts(
    texts=dokumen,
    embedding=embeddings,
    persist_directory="chroma"
)

    print("[INFO] Selesai.")
