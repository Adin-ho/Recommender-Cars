from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rag_qa import tanya_mobil

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
