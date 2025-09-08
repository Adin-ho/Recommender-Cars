# app/embedding.py
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

def ensure_chroma(csv_path: Path, persist_dir: Path, collection_name: str = "cars"):
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if coll.count() > 0:
        return coll

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    def row_text(r):
        parts = [r.get(k, "") for k in ["nama_mobil","merek","tahun","transmisi","bahan_bakar","harga"] if k in df.columns]
        return " | ".join(map(str, parts))

    docs = [row_text(r) for _, r in df.iterrows()]
    ids  = [f"id-{i}" for i in range(len(docs))]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(docs, show_progress_bar=True, batch_size=64).tolist()

    coll.add(documents=docs, embeddings=embs, ids=ids, metadatas=df.to_dict(orient="records"))
    return coll
