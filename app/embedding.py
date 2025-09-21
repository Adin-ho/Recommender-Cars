from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "app" / "data" / "data_mobil_final.csv"

def simpan_vektor_mobil(persist_dir: str):
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    req = ["Nama Mobil", "Harga", "Tahun", "Usia", "Bahan Bakar", "Transmisi", "Kapasitas Mesin"]
    for c in req:
        if c not in df.columns:
            raise RuntimeError(f"Kolom wajib hilang: {c}")

    # normalisasi
    df.columns = [c.strip() for c in df.columns]

    texts, metas = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Build texts"):
        nama = str(r["Nama Mobil"]).strip()
        tahun = int(r["Tahun"])
        harga = str(r["Harga"]).strip()
        usia = int(r["Usia"])
        bb = str(r["Bahan Bakar"]).strip()
        trans = str(r["Transmisi"]).strip()
        cc = str(r["Kapasitas Mesin"]).strip()

        text = f"{nama} tahun {tahun}. Harga {harga}. Usia {usia} tahun. Bahan bakar {bb}. Transmisi {trans}. Kapasitas mesin {cc}."
        meta = {
            "nama_mobil": nama,
            "tahun": tahun,
            "harga": harga,
            "usia": usia,
            "bahan_bakar": bb,
            "transmisi": trans,
            "kapasitas_mesin": cc,
        }
        texts.append(text)
        metas.append(meta)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("[INFO] Simpan ke:", persist_path)
    Chroma.from_texts(texts=texts, embedding=emb, metadatas=metas, persist_directory=str(persist_path))
    print("[OK] Embedding selesai. Contoh metadata:", json.dumps(metas[0], indent=2))
