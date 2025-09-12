import os
import re
import requests
from fastapi import APIRouter, Query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="/api/rag", tags=["RAG"])

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PREFER_MAX_USIA = int(os.getenv("PREFER_MAX_USIA", "5"))

# Hosted LLM (opsional)
USE_MISTRAL = bool(os.getenv("MISTRAL_API_KEY"))
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_BASE = os.getenv("MISTRAL_BASE", "https://api.mistral.ai")

USE_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

def _clean_name(nm: str) -> str:
    nm = re.sub(r"\s*\(\d{4}\)$", "", str(nm)).strip()
    nm = re.sub(r"\s{2,}", " ", nm)
    return nm

def _parse_query(q: str) -> dict:
    ql = q.lower()
    out = {"bahan_bakar": None, "transmisi": None, "usia_max": None, "tahun_min": None, "harga_max": None}

    if "diesel" in ql: out["bahan_bakar"] = "diesel"
    elif "bensin" in ql or "petrol" in ql or "gasoline" in ql: out["bahan_bakar"] = "bensin"

    if "matic" in ql or "otomatis" in ql or "automatic" in ql: out["transmisi"] = "matic"
    elif "manual" in ql: out["transmisi"] = "manual"

    m = re.search(r"(?:<=|≤|di bawah|ke\s*bawah|maks(?:imal)?)\s*(\d{1,2})\s*tahun", ql)
    if m: out["usia_max"] = int(m.group(1))

    m = re.search(r"(?:tahun\s*(?:minimal|>=|sejak)\s*)(\d{4})", ql)
    if m: out["tahun_min"] = int(m.group(1))

    m = re.search(r"(?:<=|di bawah|maks(?:imal)?)\s*([0-9\.]+)\s*(juta|jt|milyar|miliar|m)?", ql)
    if m:
        val = m.group(1).replace(".", "")
        unit = m.group(2) or ""
        n = int(val)
        if unit in ("juta", "jt", "m"): n *= 1_000_000
        elif unit in ("milyar", "miliar"): n *= 1_000_000_000
        out["harga_max"] = n
    else:
        m2 = re.search(r"rp?\s*([0-9\.]{6,})", ql)
        if m2: out["harga_max"] = int(m2.group(1).replace(".", ""))
    return out

def _passes(meta: dict, filt: dict) -> bool:
    # meta keys: nama_mobil, tahun, harga, usia, bahan_bakar, transmisi, kapasitas_mesin
    if not meta: return True
    if filt["bahan_bakar"] == "diesel" and str(meta.get("bahan_bakar","")).lower().find("diesel") < 0: return False
    if filt["bahan_bakar"] == "bensin" and re.search(r"diesel", str(meta.get("bahan_bakar","")), re.I): return False
    if filt["transmisi"] == "matic" and not re.search(r"matic|otomatis|auto", str(meta.get("transmisi","")), re.I): return False
    if filt["transmisi"] == "manual" and not re.search(r"manual", str(meta.get("transmisi","")), re.I): return False
    if filt["usia_max"] is not None and meta.get("usia") is not None and int(meta["usia"]) > filt["usia_max"]: return False
    if filt["tahun_min"] is not None and meta.get("tahun") is not None and int(meta["tahun"]) < filt["tahun_min"]: return False
    if filt["harga_max"] is not None:
        # parse harga IDR
        h = str(meta.get("harga",""))
        digits = re.findall(r"\d+", h)
        harga_num = int("".join(digits)) if digits else 0
        if harga_num > filt["harga_max"]: return False
    return True

def _mistral_chat(prompt: str, context: str) -> str | None:
    if USE_MISTRAL:
        try:
            r = requests.post(
                f"{MISTRAL_BASE}/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
                json={
                    "model": MISTRAL_MODEL,
                    "messages": [
                        {"role": "system", "content": "Jawab singkat dan relevan dengan konteks."},
                        {"role": "user", "content": f"Pertanyaan:\n{prompt}\n\nKonteks:\n{context}\n\nJawab ringkas."}
                    ],
                    "temperature": 0.2
                },
                timeout=60,
            )
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("[MISTRAL ERR]", e)

    if USE_OPENROUTER:
        try:
            r = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                         "HTTP-Referer": "https://zeabur.app", "X-Title": "car-recommender"},
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": "Jawab singkat dan relevan dengan konteks."},
                        {"role": "user", "content": f"Pertanyaan:\n{prompt}\n\nKonteks:\n{context}\n\nJawab ringkas."}
                    ],
                    "temperature": 0.2
                },
                timeout=60,
            )
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("[OPENROUTER ERR]", e)
    return None

def _get_collection():
    emb = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return Chroma(embedding_function=emb, persist_directory=PERSIST_DIR)

@router.get("", summary="RAG recommend")
def rag(pertanyaan: str = Query(..., description="Pertanyaan user"), topk: int = 5):
    filt = _parse_query(pertanyaan)
    col = _get_collection()
    # Ambil kandidat agak banyak, lalu filter & urutkan
    docs = col.similarity_search(pertanyaan, k=max(20, topk*3))

    # map ke struktur dan terapkan filter
    items = []
    for d in docs:
        m = d.metadata or {}
        items.append({
            "nama_mobil": _clean_name(m.get("nama_mobil")),
            "tahun": m.get("tahun"),
            "harga": m.get("harga"),
            "usia": m.get("usia"),
            "bahan_bakar": m.get("bahan_bakar"),
            "transmisi": m.get("transmisi"),
            "kapasitas_mesin": m.get("kapasitas_mesin"),
            "cosine_score": getattr(d, "distance", None)  # bisa None tergantung versi
        })

    # filter sesuai pertanyaan
    items = [it for it in items if _passes(it, filt)]

    # prioritas usia <= PREFER_MAX_USIA, lalu termurah, lalu termuda
    def _harga_num(h):
        if not h: return 0
        digits = re.findall(r"\d+", str(h))
        return int("".join(digits)) if digits else 0

    items.sort(key=lambda r: (
        0 if (r.get("usia") is not None and int(r["usia"]) <= PREFER_MAX_USIA) else 1,
        _harga_num(r.get("harga")),
        r.get("usia") if r.get("usia") is not None else 999
    ))

    # ambil topk
    items = items[:topk]

    if not items:
        return {"jawaban": "Tidak ditemukan.", "rekomendasi": []}

    # ringkasan untuk jawaban
    plain = "\n".join(
        f"{i}. {r['nama_mobil']}{f' ({r['tahun']})' if r.get('tahun') else ''} - {r.get('harga','-')} - "
        f"{r.get('bahan_bakar','-')}, {r.get('transmisi','-')}, {r.get('kapasitas_mesin','-')}"
        for i, r in enumerate(items, 1)
    )
    jawaban = f"Rekomendasi berdasarkan kemiripan:\n\n{plain}"

    # kalau ada API LLM → buat ringkasan natural
    llm_ans = _mistral_chat(pertanyaan, plain)
    if llm_ans:
        jawaban = llm_ans

    return {"jawaban": jawaban, "rekomendasi": items}
