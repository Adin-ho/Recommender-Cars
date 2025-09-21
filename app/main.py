# app/main.py (potongan /api/ask)
from fastapi import APIRouter, Query
from .rag_qa import detect_fuel_intent, pretty_scores, sort_young_first

router = APIRouter()

@router.get("/api/ask")
def api_ask(pertanyaan: str = Query(...), topk: int = 5):
    """
    1) Tangkap niat (diesel/listrik/hybrid).
    2) Jalankan cosine search (fungsi kamu yang sudah ada).
    3) Terapkan filter fuel bila ada niat spesifik.
    4) Urutkan usia muda dulu.
    5) Normalisasi skor agar tampil konsisten antar kategori.
    """
    q = pertanyaan.strip()
    intent = detect_fuel_intent(q)

    # ---- panggil mesin kamu (biarkan sesuai implementasi sekarang)
    # hasil = search_cosine(q, topk= max(20, topk))  # ambil agak banyak biar bisa difilter
    hasil = engine_cosine_retrieve(q, topk=max(20, topk))  # <-- ganti sesuai nama fungsi kamu
    # hasil: list[dict] dengan kunci: nama_mobil, tahun, harga, usia, bahan_bakar, transmisi, kapasitas_mesin, cosine_score

    # ---- filter ketat sesuai intent
    if intent == "diesel":
        hasil = [h for h in hasil if str(h.get("bahan_bakar","")).strip().lower() == "diesel"]
    elif intent == "listrik":
        bb = str
        hasil = [h for h in hasil if "listrik" in str(h.get("bahan_bakar","")).lower()]
    elif intent == "hybrid":
        hasil = [h for h in hasil if "hybrid" in str(h.get("bahan_bakar","")).lower()]

    # kalau setelah filter terlalu sedikit, fallback longgar (biarkan topk terpenuhi)
    if len(hasil) < topk:
        hasil = hasil[:topk]
    else:
        hasil = hasil[: max(50, topk*3)]  # pool lebih banyak → sort → ambil topk

    # ---- urutkan usia muda dulu (≤ PREFER_MAX_USIA) lalu skor
    hasil = sort_young_first(hasil)

    # ---- normalisasi skor tampil (pretty) agar tidak “terlihat kecil” di EV
    hasil = pretty_scores(hasil)

    # ambil topk terakhir
    rekom = hasil[:topk]

    judul = "Rekomendasi berdasarkan Cosine Similarity:"
    return {"jawaban": judul, "rekomendasi": rekom}
