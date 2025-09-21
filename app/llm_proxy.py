import os
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/llm", tags=["LLM"])

USE_LLM = os.getenv("ENABLE_LLM", "0") == "1"

@router.get("/echo")
def echo(q: str = Query(...)):
    # Placeholder aman. Aktifkan LLM beneran kalau sudah siap.
    if not USE_LLM:
        return {"ok": True, "answer": "(LLM dimatikan) " + q}
    # Integrasi ke Mistral/Ollama/OpenAI taruh di sini
    return {"ok": True, "answer": q}
