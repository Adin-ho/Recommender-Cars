import os
import re
from pathlib import Path
import pandas as pd

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

# ===== Paths =====
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_CSV = APP_DIR / "data" / "data_mobil_final.csv"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _harga_to_int(h):
    if pd.isna(h): return 0
    digits = re.sub(r"\D", "", str(h))
    return int(digits) if digits else 0

def _merek(nama: str) -> str:
    return str(nama).strip().split()[0].lower() if nama else ""

def _load_df() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip().str.lower()

    for col in ["nama mobil","tahun","bahan bakar","transmisi","kapasitas mesin","harga","usia"]:
        if col not in df.columns:
            df[col] = ""

    if "harga_angka" not in df.columns:
        df["harga_angka"] = df["harga"].apply(_harga_to_int)

    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").fillna(0).astype(int)
    df["usia"]  = pd.to_numeric(df["usia"],  errors="coerce").fillna(0).astype(int)
    return df

def _make_embeddings():
    # normalize_embeddings=True agar cosine murni & stabil
    return SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

def simpan_vektor_mobil():
    df = _load_df()
    embeddings = _make_embeddings()

    texts, metadatas = [], []
    for _, r in df.iterrows():
        meta = {
            "nama_mobil": str(r["nama mobil"]).strip(),
            "tahun": int(r["tahun"]),
            "harga": str(r["harga"]).strip(),
            "harga_angka": int(r["harga_angka"]),
            "usia": int(r["usia"]),
            "bahan_bakar": str(r["bahan bakar"]).strip().lower(),
            "transmisi": str(r["transmisi"]).strip().lower(),
            "kapasitas_mesin": str(r["kapasitas mesin"]).strip(),
            "merek": _merek(r["nama mobil"]),
        }
        metadatas.append(meta)
        texts.append(
            f"{r['nama mobil']} tahun {r['tahun']}, {r['bahan bakar']}, "
            f"{r['transmisi']}, kapasitas {r['kapasitas mesin']}, harga {r['harga']}"
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(CHROMA_DIR),
    )
    vs.persist()
    print("[✅ SELESAI] Embedding tersimpan di:", CHROMA_DIR)

if __name__ == "__main__":
    simpan_vektor_mobil()
