import os
import httpx
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/llm", tags=["LLM"])

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL  = os.getenv("OLLAMA_MODEL", "mistral")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))

class ChatReq(BaseModel):
    prompt: str

async def ollama_chat(prompt: str) -> str:
    """Pemanggil Ollama Chat API (non-stream)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{OLLAMA}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("message") or {}).get("content", "").strip()

@router.post("/chat")
async def chat(req: ChatReq):
    content = await ollama_chat(req.prompt)
    return {"model": MODEL, "output": content}
