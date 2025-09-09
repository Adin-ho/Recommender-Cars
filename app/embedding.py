# app/embedding.py
import os, re, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

APP_DIR   = Path(__file__).resolve().parent
ROOT_DIR  = APP_DIR.parent
DATA_CSV  = APP_DIR / "data" / "data_mobil_final.csv"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma")).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cars")

os.makedirs(CHROMA_DIR, exist_ok=True)

def _harga_to_int(s):
    if pd.isna(s): return 0
    return int(re.sub(r"\D", "", str(s)) or 0)

def simpan_vektor_mobil(collection_name: str = COLLECTION_NAME,
                        persist_dir: str = str(CHROMA_DIR)) -> None:
    """Bangun & simpan vektor ke Chroma (overwrite koleksi)."""
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    wajib = ["Nama Mobil","Harga","Tahun","Usia","Bahan Bakar","Transmisi","Kapasitas Mesin"]
    for c in wajib:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ada di CSV.")

    texts, metas = [], []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        nama  = str(r["Nama Mobil"]).strip()
        tahun = int(str(r["Tahun"]).split(".")[0]) if pd.notna(r["Tahun"]) else 0
        usia  = int(str(r["Usia"]).split(".")[0])  if pd.notna(r["Usia"])  else 0
        harga = str(r["Harga"]).strip()
        harga_angka = _harga_to_int(harga)
        bb    = str(r["Bahan Bakar"]).strip().lower()
        trans = str(r["Transmisi"]).strip().lower()
        cc    = str(r["Kapasitas Mesin"]).strip()

        texts.append(f"{nama} ({tahun}), {bb}, {trans}, harga {harga}, usia {usia} tahun, kapasitas {cc}")
        metas.append({
            "nama_mobil": nama,
            "tahun": tahun,
            "harga": harga,
            "harga_angka": harga_angka,
            "usia": usia,
            "bahan_bakar": bb,
            "transmisi": trans,
            "kapasitas_mesin": cc,
            "merek": (nama.split()[0] if nama else "").lower(),
        })

    print("[INFO] Contoh metadata:", json.dumps(metas[0], indent=2))

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    # Simpan ke koleksi tertentu
    vs = Chroma.from_texts(
        texts=texts,
        embedding=emb,
        metadatas=metas,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    vs.persist()
    try:
        count = vs._collection.count()
    except Exception:
        count = "unknown"
    print(f"[✅ SELESAI] Embedding tersimpan di {persist_dir} (collection={collection_name}, count={count})")

if __name__ == "__main__":
    simpan_vektor_mobil()
