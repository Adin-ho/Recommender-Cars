# app/main.py (hanya cuplikan endpoint)
from app.format import normalize_row

@app.get("/cosine_rekomendasi")
def cosine_rekomendasi(query: str, top_k: int = 5):
    if not query:
        return JSONResponse({"error": "query kosong"}, status_code=400)

    # 1) coba RAG
    if RAG_READY:
        try:
            from .rag_qa import cosine_rekomendasi_rag
            items = cosine_rekomendasi_rag(query=query, top_k=top_k, csv_path=DATA_CSV, persist_dir=CHROMA_DIR)
            std = []
            for it in items:
                row = normalize_row(it) if isinstance(it, dict) else normalize_row(it)
                # simpan skor kalau ada
                score = it.get("cosine_score") if isinstance(it, dict) else None
                if score is not None:
                    try: row["cosine_score"] = float(score)
                    except: pass
                std.append(row)
            return {"source": "rag", "rekomendasi": std}
        except Exception as e:
            print(f"[RAG] error: {e}")

    # 2) fallback rule-based
    try:
        from .rule_based import rekomendasi_rule_based
        items = rekomendasi_rule_based(query=query, csv_path=DATA_CSV, top_k=top_k)
        std = []
        for it in items:
            row = normalize_row(it) if isinstance(it, dict) else normalize_row(it)
            score = it.get("cosine_score") if isinstance(it, dict) else None
            if score is not None:
                try: row["cosine_score"] = float(score)
                except: pass
            std.append(row)
        return {"source": "rule_based", "rekomendasi": std}
    except Exception as e:
        return JSONResponse({"error": f"Gagal memproses query: {e}"}, status_code=400)
