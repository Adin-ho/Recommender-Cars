from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rag_qa import tanya_mobil
from app.rag_qa import stream_mobil
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API untuk rekomendasi mobil."}

@app.get("/tanya")
def tanya(pertanyaan: str):
    jawaban = tanya_mobil(pertanyaan)
    return {"jawaban": jawaban}

@app.get("/stream")
async def stream(pertanyaan: str):
    return StreamingResponse(stream_mobil(pertanyaan), media_type="text/plain")
