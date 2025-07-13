from tqdm import tqdm
import pandas as pd
import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def simpan_vektor_mobil():
    print("[INFO] Membaca dataset...")
    df = pd.read_csv("app/data/data_mobil_final.csv")
    required_cols = ['Nama Mobil', 'Harga', 'Tahun', 'Usia', 'Bahan Bakar', 'Transmisi', 'Kapasitas Mesin']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

    texts = []
    metadatas = []

    print("[INFO] Memproses baris data...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Tidak ada continue, semua baris tetap diproses!
        usia = 0
        try:
            usia = int(str(row['Usia']).strip())
        except Exception:
            pass

        harga_angka = 0
        try:
            harga_str = str(row['Harga']).replace("Rp", "").replace(".", "").replace(",", "").strip()
            harga_angka = int(harga_str)
        except Exception:
            pass

        kapasitas_mesin = str(row['Kapasitas Mesin']).strip() if 'Kapasitas Mesin' in row else "-"

        deskripsi = (
            f"{row['Nama Mobil']} ({row['Tahun']}), tahun {row['Tahun']}, harga {row['Harga']}, "
            f"usia {row['Usia']} tahun, bahan bakar {row['Bahan Bakar']}, "
            f"transmisi {row['Transmisi']}, kapasitas mesin {kapasitas_mesin}"
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
            "kapasitas_mesin": kapasitas_mesin,
        })

    print("\n[INFO] Contoh metadata:", json.dumps(metadatas[0], indent=2))
    print("[INFO] Contoh teks:", texts[0])
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
