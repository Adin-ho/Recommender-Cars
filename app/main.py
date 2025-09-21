import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import chromadb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .embedding_mistral import rebuild_index, embed_query

# ====== PATHS & ENV ======
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = str(APP_DIR / "data" / "data_mobil_final.csv")
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/data/chroma")).as_posix()

# ====== APP ======
app = FastAPI(title="ChatCars (Mistral Embeddings)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # sesuaikan di produksi
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend (opsional)
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"ok": True, "msg": "Backend up. No frontend/index.html found."}

@app.get("/health")
def health():
    return {"ok": True}

# ====== DataFrame ringan untuk metadata harga/brand (opsional filter tambahan) ======
def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip().str.lower()
    # harga angka (untuk rule filter)
    def to_int_price(x):
        if pd.isna(x): return 0
        s = str(x).lower().replace("rp", "").replace(" ", "")
        s = s.replace(",", ".")
        m = re.search(r"(\d+)\s*juta", s)
        if m: return int(m.group(1)) * 1_000_000
        digits = re.sub(r"[^\d]", "", s)
        return int(digits) if digits else 0
    if "harga_angka" not in df.columns and "harga" in df.columns:
        df["harga_angka"] = df["harga"].apply(to_int_price)
    return df

DF = load_df()

# ====== Chroma client (persist) ======
chromadb_client = chromadb.PersistentClient(path=CHROMA_DIR)
COLL = chromadb_client.get_or_create_collection("cars")

# ====== Admin: build / rebuild index dari CSV ======
@app.post("/admin/rebuild_chroma")
def admin_rebuild_chroma():
    try:
        n = rebuild_index(DATA_CSV)
        return {"ok": True, "count": n, "dir": CHROMA_DIR}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ====== Debug kecil ======
@app.get("/debug/chroma")
def debug_chroma():
    try:
        info = COLL.get(limit=5)
        return {"dir": CHROMA_DIR, "count": COLL.count(), "sample_ids": info.get("ids", [])}
    except Exception as e:
        return {"dir": CHROMA_DIR, "error": str(e)}

# ====== Search berbasis cosine (Mistral embeddings) ======
@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(query: str = Query(...), k: int = Query(5, ge=1, le=20)):
    """
    1) Embed kueri via Mistral
    2) Query ke Chroma pakai query_embeddings
    3) Kembalikan top-k + prioritas usia muda (≤ 5 tahun) dahulu
    """
    # embed kueri
    qvec = embed_query(query)

    res = COLL.query(
        query_embeddings=[qvec],
        n_results=k * 2,  # ambil lebih banyak sedikit → bisa diprioritaskan usia muda
        include=["distances", "metadatas", "documents"]
    )

    items: List[Dict[str, Any]] = []
    # jarak dari Chroma ~ (1 - cosine) karena vektor (umumnya) ter-normalisasi di sisi server
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        cosine_score = 1.0 - float(dist)
        items.append({
            "nama_mobil": meta.get("nama mobil") or meta.get("nama_mobil", ""),
            "tahun": meta.get("tahun", ""),
            "harga": meta.get("harga", ""),
            "usia": meta.get("usia", ""),
            "bahan_bakar": meta.get("bahan bakar") or meta.get("bahan_bakar",""),
            "transmisi": meta.get("transmisi",""),
            "kapasitas_mesin": meta.get("kapasitas mesin") or meta.get("kapasitas_mesin",""),
            "cosine_score": round(cosine_score, 4),
        })

    # prioritas usia muda (≤ 5 tahun)
    def usia_leq5(x) -> bool:
        try:
            u = int(x.get("usia") or 0)
        except Exception:
            # hitung dari tahun jika kolom usia tidak ada
            try:
                th = int(x.get("tahun") or 0)
                u = max(0, 2025 - th) if th else 999
            except Exception:
                u = 999
        return u <= 5

    muda = [x for x in items if usia_leq5(x)]
    tua = [x for x in items if not usia_leq5(x)]
    hasil = (muda + tua)[:k]

    return {"rekomendasi": hasil}
