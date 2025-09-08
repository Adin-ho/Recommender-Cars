# app/rag_qa.py
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

def cosine_rekomendasi_rag(query: str, top_k: int, csv_path: Path, persist_dir: Path) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(Path(persist_dir)))
    coll = client.get_or_create_collection(name="cars", metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query]).tolist()

    out = coll.query(query_embeddings=q_emb, n_results=top_k, include=["metadatas","distances"])
    metas = out.get("metadatas", [[]])[0]
    dists = out.get("distances", [[]])[0]
    results = []
    for m, d in zip(metas, dists):
        item = dict(m) if isinstance(m, dict) else {}
        item["score"] = 1 - float(d)  # approx cosine similarity
        results.append(item)
    return results
