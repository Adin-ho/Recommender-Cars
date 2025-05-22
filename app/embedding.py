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
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

    texts = []
    metadatas = []

    print("[INFO] Memproses baris data...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Ambil dan bersihkan usia
            usia = int(str(row['Usia']).strip())

            # Konversi harga string ke angka (int)
            harga_str = str(row['Harga']).replace("Rp", "").replace(".", "").replace(",", "").strip()
            harga_angka = int(harga_str) if harga_str.isdigit() else 0

            deskripsi = (
                f"{row['Nama Mobil']} ({row['Tahun']}), tahun {row['Tahun']}, harga {row['Harga']}, "
                f"usia {row['Usia']} tahun, bahan bakar {row['Bahan Bakar']}, "
                f"transmisi {row['Transmisi']}, kapasitas mesin {row['Kapasitas Mesin']} cc"
            )
            texts.append(deskripsi)

            metadatas.append({
                "nama_mobil": str(row['Nama Mobil']).strip(),
                "tahun": int(row['Tahun']),
                "harga": str(row['Harga']).strip(),
                "harga_angka": harga_angka,
                "usia": usia,
                "bahan_bakar": str(row['Bahan Bakar']).strip().lower(),
                "transmisi": str(row['Transmisi']).strip().lower(),
            })

        except Exception as e:
            print(f"[SKIP] Baris dilewati karena error: {e}")
            continue

    embeddings = OllamaEmbeddings(model="mistral")

    print("[INFO] Menyimpan ke ChromaDB...")
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="chroma"
    )

    print("[✅ SELESAI] Semua data valid disimpan ke ChromaDB.")

if __name__ == "__main__":
    simpan_vektor_mobil()
