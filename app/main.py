# app/main.py
from fastapi import FastAPI
from app.rag_qa import tanya_mobil

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API untuk rekomendasi mobil berdasarkan pertanyaan."}

@app.get("/tanya")
def tanya(pertanyaan: str):
    jawaban = tanya_mobil(pertanyaan)
    return {"jawaban": jawaban}
