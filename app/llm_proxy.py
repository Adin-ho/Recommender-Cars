import os, httpx
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/llm", tags=["LLM"])
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL  = os.getenv("OLLAMA_MODEL", "mistral")

class ChatReq(BaseModel):
    prompt: str

@router.post("/chat")
async def chat(req: ChatReq):
    payload = {
        "model": MODEL,
        "messages": [{"role":"user","content": req.prompt}],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{OLLAMA}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    # format keluaran ringkas
    content = data.get("message",{}).get("content","")
    return {"model": MODEL, "output": content}
