import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
import time

def simpan_vektor_mobil():
    print("[INFO] Membaca dataset...")
    df = pd.read_csv("app/data/data_mobil.csv")

    required_cols = ['Nama Mobil', 'Harga', 'Tahun', 'Usia', 'Bahan Bakar', 'Transmisi', 'Kapasitas Mesin']
    print("[DEBUG] Kolom tersedia:", list(df.columns))

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

    print("[INFO] Mulai proses embedding...")
    dokumen = [
        f"{row['Nama Mobil']} ({row['Tahun']}), usia {row['Usia']} tahun, "
        f"harga {row['Harga']}, transmisi {row['Transmisi']}, bahan bakar {row['Bahan Bakar']}, "
        f"kapasitas mesin {row['Kapasitas Mesin']} cc"
        for _, row in tqdm(df.iterrows(), total=len(df))
    ]

    embeddings = OllamaEmbeddings(model="mistral")

    print("[INFO] Menyimpan ke ChromaDB dalam batch...")
    start = time.time()
    Chroma.from_texts(
        texts=dokumen,
        embedding=embeddings,
        persist_directory="chroma"
    )
    end = time.time()

    print(f"[INFO] Selesai simpan ke ChromaDB. Waktu: {end - start:.2f} detik.")

if __name__ == "__main__":
    simpan_vektor_mobil()
