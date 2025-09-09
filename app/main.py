import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ====== Path dasar ======
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))
ENABLE_RAG = os.getenv("ENABLE_RAG", "0") == "1"

# ====== Aplikasi ======
app = FastAPI(title="ChatCars")

# CORS (boleh disesuaikan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti ke domain kamu kalau mau lebih ketat
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Serve frontend ======
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health():
    return {"ok": True}

# ====== Dataset mobil ======
def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip().str.lower()
    # normalisasi kolom harga -> angka
    if "harga_angka" not in df.columns:
        def _to_int(x):
            s = str(x)
            digits = re.sub(r"\D", "", s)
            return int(digits) if digits else 0
        if "harga" in df.columns:
            df["harga_angka"] = df["harga"].apply(_to_int)
        else:
            df["harga_angka"] = 0
    return df

data_mobil = _load_dataset()

# ====== Rule-based filter sederhana (backup kalau RAG kosong) ======
def filter_rule_based(q: str, limit: int = 5) -> pd.DataFrame:
    qlow = q.lower()
    df = data_mobil.copy()

    # Bahan bakar
    fuels = ["diesel", "bensin", "hybrid", "listrik"]
    for bb in fuels:
        if bb in qlow and f"non {bb}" not in qlow:
            if "bahan bakar" in df.columns:
                df = df[df["bahan bakar"].str.contains(bb, case=False, na=False)]

    # Transmisi
    if "matic" in qlow and "manual" not in qlow and "transmisi" in df.columns:
        df = df[df["transmisi"].str.contains("matic", case=False, na=False)]
    if "manual" in qlow and "matic" not in qlow and "transmisi" in df.columns:
        df = df[df["transmisi"].str.contains("manual", case=False, na=False)]

    # Harga
    m_max = re.search(r"(?:di\s*bawah|max(?:imal)?|<=?)\s*rp?\s*([\d\.]+)", qlow)
    if m_max:
        batas = int(m_max.group(1).replace(".", ""))
        df = df[df["harga_angka"] <= batas]
    m_range = re.search(r"rp?\s*([\d\.]+)\s*-\s*rp?\s*([\d\.]+)", qlow)
    if m_range:
        lo = int(m_range.group(1).replace(".", ""))
        hi = int(m_range.group(2).replace(".", ""))
        df = df[(df["harga_angka"] >= lo) & (df["harga_angka"] <= hi)]

    # Tahun ke atas
    m_tahun_min = re.search(r"tahun\s*(\d{4})\s*(?:ke\s*atas|\+)", qlow)
    if m_tahun_min and "tahun" in df.columns:
        df = df[df["tahun"] >= int(m_tahun_min.group(1))]

    if df.empty:
        return df
    return df.head(limit)

# ====== (Opsional) RAG: pakai Chroma + Sentence Transformers ======
if ENABLE_RAG:
    try:
        from app.rag_qa import build_retriever, rebuild_chroma as _rebuild
        retriever = build_retriever(CHROMA_DIR, DATA_CSV)

        @app.post("/admin/rebuild_chroma")
        def rebuild_chroma():
            _rebuild(CHROMA_DIR, DATA_CSV)
            return JSONResponse({"ok": True, "dir": str(CHROMA_DIR)})

        @app.get("/debug/chroma")
        def debug_chroma():
            files = []
            if CHROMA_DIR.exists():
                files = [p.name for p in CHROMA_DIR.glob("*")]
            return {"dir": str(CHROMA_DIR), "files": files}

        @app.get("/debug/chroma_count")
        def debug_chroma_count():
            try:
                results = retriever.vectorstore._collection.count()  # type: ignore
            except Exception:
                results = None
            return {"count": results}

    except Exception as e:
        # Jangan biarkan startup mati hanya karena RAG error
        print("[INIT] ENABLE_RAG=1 tapi gagal inisialisasi RAG:", e)
        ENABLE_RAG = False

# ====== Endpoint yang dipakai front-end ======
@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(query: str, k: int = 5) -> Dict[str, Any]:
    """
    1) Coba cari via RAG (semantic search)
    2) Jika kosong, fallback rule-based filter
    """
    results: List[Dict[str, Any]] = []

    # 1) RAG
    if ENABLE_RAG:
        try:
            docs = retriever.get_relevant_documents(query)[:k]  # type: ignore
            for d in docs:
                meta = d.metadata or {}
                results.append({
                    "nama_mobil": meta.get("nama mobil") or meta.get("nama") or "",
                    "tahun": meta.get("tahun") or "",
                    "harga": meta.get("harga") or "",
                    "usia": meta.get("usia") or "",
                    "bahan_bakar": meta.get("bahan bakar") or meta.get("bahan_bakar") or "",
                    "transmisi": meta.get("transmisi") or "",
                    "kapasitas_mesin": meta.get("kapasitas mesin") or meta.get("kapasitas_mesin") or "",
                    "cosine_score": getattr(d, "score", None)
                })
        except Exception as e:
            print("[RAG] gagal search:", e)

    # 2) Fallback rule-based
    if not results:
        df = filter_rule_based(query, k)
        for _, r in df.iterrows():
            results.append({
                "nama_mobil": r.get("nama mobil", ""),
                "tahun": r.get("tahun", ""),
                "harga": r.get("harga", ""),
                "usia": r.get("usia", ""),
                "bahan_bakar": r.get("bahan bakar", ""),
                "transmisi": r.get("transmisi", ""),
                "kapasitas_mesin": r.get("kapasitas mesin", ""),
                "cosine_score": None
            })

    return {"rekomendasi": results}
