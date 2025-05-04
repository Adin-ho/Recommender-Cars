import pandas as pd
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

def simpan_vektor_mobil():
    df = pd.read_csv("app/data/data_mobil.csv")
    embeddings = OllamaEmbeddings(model="mistral")

    # Format kalimat per mobil
    dokumen = [
        f"{row['Nama Mobil']} ({row['Tahun']}), usia {row['Usia Kendaraan (tahun)']} tahun, "
        f"harga {row['Harga']}, transmisi {row['Transmisi']}, bahan bakar {row['Bahan Bakar']}, "
        f"kapasitas mesin {row['Kapasitas Mesin (cc)']} cc"
        for _, row in df.iterrows()
    ]

    # Simpan ke ChromaDB
    Chroma.from_texts(
        texts=dokumen,
        embedding=embeddings,
        persist_directory="chroma"
    ).persist()
