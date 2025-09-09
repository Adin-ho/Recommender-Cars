import os, re, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_CSV   = ROOT_DIR / "app" / "data" / "data_mobil_final.csv"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _harga_int(s):
    if pd.isna(s): return 0
    m = re.sub(r"\D", "", str(s))
    return int(m) if m else 0

def simpan_vektor_mobil(collection_name: str = "cars", persist_dir: str = "chroma"):
    print("[INFO] Membaca:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    vectordb = Chroma(collection_name=collection_name, persist_directory=persist_dir, embedding_function=emb)
    wajib = ['Nama Mobil','Harga','Tahun','Usia','Bahan Bakar','Transmisi','Kapasitas Mesin']
    for c in wajib:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan di CSV.")

    texts, metas = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        nama  = str(row['Nama Mobil']).strip()
        tahun = int(row['Tahun'])
        harga = str(row['Harga']).strip()
        harga_angka = _harga_int(harga)
        usia  = int(str(row['Usia']).split(".")[0]) if pd.notna(row['Usia']) else 0
        bb    = str(row['Bahan Bakar']).strip().lower()
        trans = str(row['Transmisi']).strip().lower()
        cc    = str(row.get('Kapasitas Mesin','-') or '-').strip()

        texts.append(f"{nama} ({tahun}), {bb}, {trans}, harga {harga}, usia {usia} tahun, kapasitas {cc}")
        metas.append({
            "nama_mobil": nama, "tahun": tahun, "harga": harga,
            "harga_angka": harga_angka, "usia": usia,
            "bahan_bakar": bb, "transmisi": trans, "kapasitas_mesin": cc,
            "merek": (nama.split()[0] if nama else "").lower(),
        })

    print("[INFO] Contoh metadata:", json.dumps(metas[0], indent=2))
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                encode_kwargs={"normalize_embeddings": True})
    vs = Chroma.from_texts(texts=texts, embedding=emb, metadatas=metas, persist_directory=str(CHROMA_DIR))
    vs.persist()
    try: count = vs._collection.count()
    except: count = "unknown"
    print(f"[✅ SELESAI] Embedding tersimpan di {CHROMA_DIR} (count={count})")

if __name__ == "__main__":
    simpan_vektor_mobil()
