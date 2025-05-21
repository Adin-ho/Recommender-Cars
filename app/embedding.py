import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
import time
import json

def simpan_vektor_mobil():
    print("[INFO] Membaca dataset...")
    df = pd.read_csv("app/data/data_mobil.csv")

    required_cols = ['Nama Mobil', 'Harga', 'Tahun', 'Usia', 'Bahan Bakar', 'Transmisi', 'Kapasitas Mesin']
    print("[DEBUG] Kolom tersedia:", list(df.columns))

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

    print("[INFO] Mulai proses embedding...")

    texts = []
    metadatas = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        deskripsi = (
            f"{row['Nama Mobil']} ({row['Tahun']}), tahun {row['Tahun']}, harga {row['Harga']}, "
            f"usia {row['Usia']} tahun, bahan bakar {row['Bahan Bakar']}, "
            f"transmisi {row['Transmisi']}, kapasitas mesin {row['Kapasitas Mesin']} cc"
        )
        texts.append(deskripsi)

        # Simpan metadata dalam format lowercase agar bisa di-query dengan .lower()
        metadatas.append({
            "nama_mobil": str(row['Nama Mobil']).strip(),
            "tahun": str(row['Tahun']),
            "harga": str(row['Harga']),
            "usia": str(row['Usia']),
            "bahan_bakar": str(row['Bahan Bakar']).strip().lower(),
            "transmisi": str(row['Transmisi']).strip().lower(),
        })

    embeddings = OllamaEmbeddings(model="mistral")

    print("[INFO] Menyimpan ke ChromaDB...")
    start = time.time()
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="chroma"
    )
    end = time.time()

    print(f"[INFO] Selesai simpan ke ChromaDB. Waktu: {end - start:.2f} detik.")

    # Simpan versi mentah (opsional) untuk validasi/debugging
    with open("app/data/mobil_data.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    simpan_vektor_mobil()
