import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
import time
import json

def simpan_vektor_mobil():
    print("[INFO] Membaca dataset...")
    df = pd.read_csv("data/data_mobil_final.csv")

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
            try:
                harga_angka = int(harga_str)
            except:
                harga_angka = 0


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
                "harga_angka": int(row['harga_angka']) if 'harga_angka' in row and pd.notna(row['harga_angka']) else harga_angka,
                "usia": usia,
                "bahan_bakar": str(row['Bahan Bakar']).strip().lower(),
                "transmisi": str(row['Transmisi']).strip().lower(),
                "kapasitas_mesin": str(row['Kapasitas Mesin']).strip() if 'Kapasitas Mesin' in row else None,
            })

        except Exception as e:
            print(f"[SKIP] Baris dilewati karena error: {e}")
            continue

    print("\n[DEBUG] Contoh metadata:", json.dumps(metadatas[0], indent=2))
    print("[DEBUG] Contoh teks:", texts[0])
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
