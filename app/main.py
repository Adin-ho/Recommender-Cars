from fastapi import FastAPI, Query
from app.rag_qa import jawab_pertanyaan

app = FastAPI()

@app.get("/rekomendasi")
def rekomendasi(q: str = Query(..., description="Masukkan preferensi mobil")):
    hasil = jawab_pertanyaan(q)
    return hasil
