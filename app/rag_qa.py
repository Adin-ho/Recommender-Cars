from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Model ringan CPU
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def _load_df(data_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(data_csv)
    df.columns = df.columns.str.strip().str.lower()
    # normalisasi usia jika belum ada
    if "usia" not in df.columns and "tahun" in df.columns:
        df["usia"] = df["tahun"].apply(lambda t: "" if pd.isna(t) else max(0, 2025 - int(t)))
    return df

def rebuild_chroma(chroma_dir: Path, data_csv: Path) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)

    df = _load_df(data_csv)

    # gabung atribut jadi satu “dokumen”
    def as_doc(row) -> str:
        parts = [
            str(row.get("nama mobil", "")),
            f"tahun {row.get('tahun', '')}",
            f"harga {row.get('harga', '')}",
            f"usia {row.get('usia', '')} tahun",
            f"bahan bakar {row.get('bahan bakar', '')}",
            f"transmisi {row.get('transmisi', '')}",
            f"kapasitas mesin {row.get('kapasitas mesin', '')}",
        ]
        return " | ".join([p for p in parts if p])

    docs = [as_doc(r) for _, r in df.iterrows()]
    metadatas = [r.to_dict() for _, r in df.iterrows()]
    ids = [f"car-{i}" for i in range(len(docs))]

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    # hapus koleksi lama kalau ada
    try:
        Chroma(
            collection_name="cars",
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        ).delete_collection()
    except Exception:
        pass

    vs = Chroma.from_texts(
        texts=docs,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name="cars",
        persist_directory=str(chroma_dir),
    )
    vs.persist()

def build_retriever(chroma_dir: Path, data_csv: Path):
    chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # Kalau index belum ada, bikin dulu
    need_build = True
    for p in chroma_dir.glob("*"):
        need_build = False
        break
    if need_build:
        rebuild_chroma(chroma_dir, data_csv)

    vs = Chroma(
        collection_name="cars",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    return vs.as_retriever(search_kwargs={"k": 5})
