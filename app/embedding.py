from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import chromadb


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT_DIR / "app" / "data" / "data_mobil_final.csv"
CHROMA_DIR = ROOT_DIR / "chroma"

def ensure_chroma(csv_path: Path, persist_dir: Path, collection_name: str = "cars"):
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if coll.count() > 0:
        return coll

    df = pd.read_csv(csv_path)
    texts = [
        " | ".join(map(str, [
            row.get("nama_mobil",""),
            row.get("merek",""),
            row.get("tahun",""),
            row.get("transmisi",""),
            row.get("bahan_bakar",""),
            row.get("harga",""),
        ]))
        for _, row in df.iterrows()
    ]
    ids = [f"id-{i}" for i in range(len(texts))]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, show_progress_bar=True, batch_size=64).tolist()
    coll.add(documents=texts, embeddings=embs, ids=ids, metadatas=df.to_dict(orient="records"))
    return coll

def simpan_vektor_mobil():
    print("[INFO] Membaca dataset:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)

    required_cols = ['Nama Mobil', 'Harga', 'Tahun', 'Usia', 'Bahan Bakar', 'Transmisi', 'Kapasitas Mesin']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

    texts, metadatas = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            usia = int(str(row['Usia']).strip())
        except Exception:
            usia = 0

        try:
            harga_str = str(row['Harga']).replace("Rp", "").replace(".", "").replace(",", "").strip()
            harga_angka = int(harga_str)
        except Exception:
            harga_angka = 0

        kapasitas = str(row.get('Kapasitas Mesin', '-') or '-').strip()

        deskripsi = (
            f"{row['Nama Mobil']} ({row['Tahun']}), tahun {row['Tahun']}, harga {row['Harga']}, "
            f"usia {row['Usia']} tahun, bahan bakar {row['Bahan Bakar']}, "
            f"transmisi {row['Transmisi']}, kapasitas mesin {kapasitas}"
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
            "kapasitas_mesin": kapasitas,
        })

    print("[INFO] Contoh metadata:", json.dumps(metadatas[0], indent=2))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("[INFO] Menyimpan ke ChromaDB:", CHROMA_DIR)
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(CHROMA_DIR)
    )
    print("[✅ SELESAI] Embedding tersimpan.")

if __name__ == "__main__":
    simpan_vektor_mobil()
