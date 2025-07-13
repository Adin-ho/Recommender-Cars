import random
import re
from fastapi import APIRouter, Query
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

router = APIRouter()

def valid_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default

def is_kapasitas_mesin_valid(kapasitas, bahan_bakar):
    # Listrik/hybrid tidak dicek CC, yang penting stringnya ada (kWh, dsb)
    if "listrik" in str(bahan_bakar).lower() or "hybrid" in str(bahan_bakar).lower():
        return kapasitas is not None and len(str(kapasitas).strip()) > 0
    try:
        num = int(re.sub(r'\D', '', str(kapasitas)))
        return 600 <= num <= 6000
    except:
        return False

@router.get("/cosine_rekomendasi")
async def cosine_rekomendasi(
    query: str = Query(..., description="Pertanyaan/kebutuhan mobil, misal 'rekomendasi mobil 500 juta'"),
    k: int = Query(5, description="Jumlah hasil yang ingin ditampilkan"),
    exclude: str = Query("", description="Nama mobil yang sudah direkomendasikan, pisahkan koma")
):
    embeddings = OllamaEmbeddings(model="mistral")
    vector_store = Chroma(
        persist_directory="chroma",
        embedding_function=embeddings,
    )
    # Ambil banyak, supaya random benar-benar dinamis (jika exclude dipakai)
    result_list = vector_store.similarity_search_with_score(query, k=150)

    q_lc = query.lower()
    harga_target = None
    match_harga = re.search(r"(\d{2,4})\s*juta", q_lc)
    if match_harga:
        harga_target = int(match_harga.group(1)) * 1_000_000
    else:
        match_angka = re.search(r"(\d{9,12})", q_lc.replace(".", ""))
        if match_angka:
            harga_target = int(match_angka.group(1))
    tolerance = 0.18  # ±18%
    harga_min, harga_max = 0, 10**10
    if harga_target:
        harga_min = int(harga_target * (1 - tolerance))
        harga_max = int(harga_target * (1 + tolerance))
    usia_max = 5
    match_usia = re.search(r"(?:<|di ?bawah|kurang dari|maks(?:imal)?|max(?:imum)?)\s*(\d{1,2})\s*tahun", q_lc)
    if match_usia:
        usia_max = int(match_usia.group(1))

    # Deteksi bahan bakar secara cerdas
    filter_bahan_bakar = None
    for bb in ["listrik", "electric", "diesel", "bensin", "hybrid"]:
        if bb in q_lc:
            filter_bahan_bakar = bb
            break

    exclude_list = [x.strip().lower() for x in exclude.split(",") if x.strip()]
    hasil_utama, hasil_tua, hasil_lain = [], [], []

    # Track mobil yang sudah diambil, agar tidak dobel
    seen = set()

    for doc, score in result_list:
        meta = doc.metadata
        nama = str(meta.get("nama_mobil", "-"))
        harga = valid_int(meta.get("harga_angka", 0))
        harga_display = meta.get("harga", "-")
        usia = valid_int(meta.get("usia", 0))
        kapasitas = meta.get("kapasitas_mesin", "-")
        bb = str(meta.get("bahan_bakar", "-")).lower()
        trans = str(meta.get("transmisi", "-"))
        tahun = meta.get("tahun", "-")

        # Exclude & deduplicate
        key_nama = f"{nama.lower().strip()}__{tahun}"
        if nama.lower() in exclude_list or key_nama in seen:
            continue
        seen.add(key_nama)

        # Bahan bakar filter
        if filter_bahan_bakar and filter_bahan_bakar not in bb:
            continue

        # Validasi kapasitas mesin (khusus hybrid/listrik true jika ada apapun)
        if not is_kapasitas_mesin_valid(kapasitas, bb):
            kapasitas = "-"

        data_obj = {
            "nama_mobil": nama,
            "tahun": tahun,
            "harga": harga_display,
            "harga_angka": harga,
            "usia": usia,
            "bahan_bakar": bb,
            "transmisi": trans,
            "kapasitas_mesin": kapasitas,
            "cosine_score": float(round(float(score), 4))
        }

        # Prioritas: harga sesuai ±18%, usia < 5 tahun
        if harga_target and (harga_min <= harga <= harga_max):
            if 0 < usia <= usia_max:
                hasil_utama.append(data_obj)
            elif usia > usia_max:
                hasil_tua.append(data_obj)
        elif not harga_target and 0 < usia <= usia_max:
            hasil_utama.append(data_obj)
        else:
            hasil_lain.append(data_obj)

    # Sorting hasil utama (dekat harga & usia muda), hasil_tua (usia tua tapi harga dekat)
    hasil_utama = sorted(hasil_utama, key=lambda x: (abs(x['harga_angka']-harga_target) if harga_target else x['usia'], x['usia']))
    hasil_tua = sorted(hasil_tua, key=lambda x: (x['usia'], abs(x['harga_angka']-harga_target) if harga_target else 0))
    random.shuffle(hasil_lain)

    hasil_final = hasil_utama + hasil_tua + hasil_lain
    hasil_final = hasil_final[:k]

    # Jika kosong, tetap tampilkan hasil random tanpa filter (alternatif)
    if not hasil_final and result_list:
        alt = []
        seen_alt = set()
        for doc, score in result_list:
            meta = doc.metadata
            nama = str(meta.get("nama_mobil", "-"))
            tahun = meta.get("tahun", "-")
            key_nama = f"{nama.lower().strip()}__{tahun}"
            if nama.lower() in exclude_list or key_nama in seen_alt:
                continue
            seen_alt.add(key_nama)
            alt.append({
                "nama_mobil": nama,
                "tahun": tahun,
                "harga": meta.get("harga", "-"),
                "harga_angka": valid_int(meta.get("harga_angka", 0)),
                "usia": valid_int(meta.get("usia", 0)),
                "bahan_bakar": str(meta.get("bahan_bakar", "-")).lower(),
                "transmisi": str(meta.get("transmisi", "-")),
                "kapasitas_mesin": meta.get("kapasitas_mesin", "-"),
                "cosine_score": float(round(float(score), 4))
            })
            if len(alt) >= k:
                break
        hasil_final = alt

    if not hasil_final:
        return {
            "jawaban": "Maaf, tidak ditemukan mobil yang sesuai dengan kriteria pencarian Anda.",
            "rekomendasi": []
        }

    # Output Format
    output = "Rekomendasi berdasarkan Cosine Similarity:\n\n"
    for idx, m in enumerate(hasil_final, 1):
        output += (
            f"{idx}. {m['nama_mobil']} ({m['tahun']})\n"
            f"    Skor: {m['cosine_score']}\n"
            f"    Harga: {m['harga']}\n"
            f"    Usia: {m['usia']} tahun\n"
            f"    Bahan Bakar: {m['bahan_bakar'].capitalize()}\n"
            f"    Transmisi: {m['transmisi'].capitalize()}\n"
            f"    Kapasitas Mesin: {m['kapasitas_mesin']}\n\n"
        )
    return {"jawaban": output, "rekomendasi": hasil_final}
