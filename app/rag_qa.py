import os
import re
import random
from fastapi import APIRouter, Query
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # pastikan tidak pakai GPU/CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PERSIST_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).resolve().parents[1] / "chroma"))
# ===== Pilih embedding: Ollama (kalau ada) atau CPU (default) =====
if os.getenv("USE_OLLAMA", "0") == "1":
    from langchain_ollama import OllamaEmbeddings as _Emb
    EMBEDDINGS = _Emb(model="mistral")
else:
    # CPU: ringan & cocok free hosting
    from langchain_community.embeddings import HuggingFaceEmbeddings as _Emb
    EMBEDDINGS = _Emb(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_chroma import Chroma

router = APIRouter()

def _parse_filter(q: str):
    ql = q.lower()
    where = {}
    if any(k in ql for k in ["listrik", "ev", "electric"]):
        where["bahan_bakar"] = "listrik"
    elif "diesel" in ql:
        where["bahan_bakar"] = "diesel"
    elif "bensin" in ql:
        where["bahan_bakar"] = "bensin"
    # merek umum
    for brand in ["bmw","toyota","honda","suzuki","renault","wuling","mitsubishi","daihatsu","nissan","mazda","hyundai","kia","vw","volkswagen"]:
        if brand in ql:
            where["merek"] = brand
            break
    return where

def valid_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def is_kapasitas_mesin_valid(kapasitas, bahan_bakar):
    j = str(bahan_bakar).lower()
    if "listrik" in j or "hybrid" in j:
        return kapasitas is not None and str(kapasitas).strip() != ""
    try:
        num = int(re.sub(r"\D", "", str(kapasitas)))
        return 600 <= num <= 6000
    except Exception:
        return False

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(
    query: str = Query(..., description="Pertanyaan kebutuhan mobil (mis. 'mpv 200 juta')"),
    k: int = Query(5, description="Jumlah hasil"),
    exclude: str = Query("", description="Nama mobil yang sudah direkomendasikan, pisahkan koma")
):
    vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=EMBEDDINGS,
)
    result_list = vector_store.similarity_search_with_score(query, k=150)

    q_lc = query.lower()

    # Target harga (boleh '200 juta' atau angka utuh)
    harga_target = None
    m = re.search(r"(\d{2,4})\s*juta", q_lc)
    if m:
        harga_target = int(m.group(1)) * 1_000_000
    else:
        m2 = re.search(r"(\d{9,12})", q_lc.replace(".", ""))
        if m2:
            harga_target = int(m2.group(1))

    tolerance = 0.18
    harga_min, harga_max = 0, 10**10
    if harga_target:
        harga_min = int(harga_target * (1 - tolerance))
        harga_max = int(harga_target * (1 + tolerance))

    # Usia maks (default 5 thn)
    usia_max = 5
    m_usia = re.search(r"(?:<|di ?bawah|kurang dari|max(?:imal)?)\s*(\d{1,2})\s*tahun", q_lc)
    if m_usia:
        usia_max = int(m_usia.group(1))

    # Filter bahan bakar (opsional)
    filter_bb = None
    for bb in ["listrik", "electric", "diesel", "bensin", "hybrid"]:
        if bb in q_lc:
            filter_bb = bb
            break

    exclude_list = [x.strip().lower() for x in exclude.split(",") if x.strip()]
    hasil_utama, hasil_tua, hasil_lain = [], [], []
    seen = set()

    for doc, score in result_list:
        meta = doc.metadata
        nama = str(meta.get("nama_mobil", "-"))
        tahun = meta.get("tahun", "-")
        harga = valid_int(meta.get("harga_angka", 0))
        harga_disp = meta.get("harga", "-")
        usia = valid_int(meta.get("usia", 0))
        bb = str(meta.get("bahan_bakar", "-")).lower()
        trans = str(meta.get("transmisi", "-"))
        kapasitas = meta.get("kapasitas_mesin", "-")

        key = f"{nama.lower().strip()}__{tahun}"
        if nama.lower() in exclude_list or key in seen:
            continue
        seen.add(key)

        if filter_bb and filter_bb not in bb:
            continue

        if not is_kapasitas_mesin_valid(kapasitas, bb):
            kapasitas = "-"

        obj = {
            "nama_mobil": nama,
            "tahun": tahun,
            "harga": harga_disp,
            "harga_angka": harga,
            "usia": usia,
            "bahan_bakar": bb,
            "transmisi": trans,
            "kapasitas_mesin": kapasitas,
            "cosine_score": float(round(float(score), 4)),
        }

        if harga_target and (harga_min <= harga <= harga_max):
            if 0 < usia <= usia_max:
                hasil_utama.append(obj)
            elif usia > usia_max:
                hasil_tua.append(obj)
        elif not harga_target and 0 < usia <= usia_max:
            hasil_utama.append(obj)
        else:
            hasil_lain.append(obj)

    hasil_utama = sorted(hasil_utama, key=lambda x: (abs(x['harga_angka']-(harga_target or 0)), x['usia']))
    hasil_tua   = sorted(hasil_tua,   key=lambda x: (x['usia'], abs(x['harga_angka']-(harga_target or 0))))
    random.shuffle(hasil_lain)

    hasil_final = (hasil_utama + hasil_tua + hasil_lain)[:k]

    if not hasil_final and result_list:
        alt, seen_alt = [], set()
        for doc, score in result_list:
            meta = doc.metadata
            nama = str(meta.get("nama_mobil", "-"))
            tahun = meta.get("tahun", "-")
            key = f"{nama.lower().strip()}__{tahun}"
            if nama.lower() in exclude_list or key in seen_alt:
                continue
            seen_alt.add(key)
            alt.append({
                "nama_mobil": nama,
                "tahun": tahun,
                "harga": meta.get("harga", "-"),
                "harga_angka": valid_int(meta.get("harga_angka", 0)),
                "usia": valid_int(meta.get("usia", 0)),
                "bahan_bakar": str(meta.get("bahan_bakar", "-")).lower(),
                "transmisi": str(meta.get("transmisi", "-")),
                "kapasitas_mesin": meta.get("kapasitas_mesin", "-"),
                "cosine_score": float(round(float(score), 4)),
            })
            if len(alt) >= k:
                break
        hasil_final = alt

    if not hasil_final:
        return {"jawaban": "Maaf, tidak ditemukan mobil yang sesuai.", "rekomendasi": []}

    out = "Rekomendasi berdasarkan Cosine Similarity:\n\n"
    for i, m in enumerate(hasil_final, 1):
        out += (
            f"{i}. {m['nama_mobil']} ({m['tahun']})\n"
            f"    Skor: {m['cosine_score']}\n"
            f"    Harga: {m['harga']}\n"
            f"    Usia: {m['usia']} tahun\n"
            f"    Bahan Bakar: {m['bahan_bakar'].capitalize()}\n"
            f"    Transmisi: {m['transmisi'].capitalize()}\n"
            f"    Kapasitas Mesin: {m['kapasitas_mesin']}\n\n"
        )
    return {"jawaban": out, "rekomendasi": hasil_final}
