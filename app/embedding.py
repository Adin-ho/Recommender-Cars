import os
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ===== Paths =====
APP_DIR   = Path(__file__).resolve().parent
ROOT_DIR  = APP_DIR.parent
DATA_CSV  = APP_DIR / "data" / "data_mobil_final.csv"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _harga_to_int(s):
    if pd.isna(s): return 0
    return int(re.sub(r"\D", "", str(s)) or 0)

def simpan_vektor_mobil():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    # dataset kamu pakai huruf besar pada header → jangan di-lower semua
    req = ["Nama Mobil","Harga","Tahun","Usia","Bahan Bakar","Transmisi","Kapasitas Mesin"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan di CSV.")

    texts, metadatas = [], []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        nama   = str(r["Nama Mobil"]).strip()
        tahun  = int(str(r["Tahun"]).split(".")[0]) if pd.notna(r["Tahun"]) else 0
        usia   = int(str(r["Usia"]).split(".")[0])  if pd.notna(r["Usia"])  else 0
        harga  = str(r["Harga"]).strip()
        h_angka= _harga_to_int(harga)
        bb     = str(r["Bahan Bakar"]).strip().lower()
        trans  = str(r["Transmisi"]).strip().lower()
        cc     = str(r["Kapasitas Mesin"]).strip()

        texts.append(f"{nama} ({tahun}), {bb}, {trans}, harga {harga}, usia {usia} tahun, kapasitas {cc}")
        metadatas.append({
            "nama_mobil": nama,
            "tahun": tahun,
            "harga": harga,
            "harga_angka": h_angka,
            "usia": usia,
            "bahan_bakar": bb,
            "transmisi": trans,
            "kapasitas_mesin": cc,
            "merek": (nama.split()[0] if nama else "").lower()
        })

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    # BUAT koleksi & PERSIST dengan pasti
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(CHROMA_DIR),
    )
    vs.persist()  # penting
    # Log jumlah dokumen
    try:
        count = vs._collection.count()
    except Exception:
        count = "unknown"
    print(f"[✅ SELESAI] Embedding tersimpan di: {CHROMA_DIR} (count={count})")

if __name__ == "__main__":
    simpan_vektor_mobil()
