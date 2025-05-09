from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rag_qa import stream_mobil
from fastapi.responses import StreamingResponse

app = FastAPI()

# Izinkan frontend lokal mengakses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # sesuaikan jika frontend beda host/port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API ChatCars aktif."}

@app.get("/stream")
async def stream(pertanyaan: str):
    return await stream_mobil(pertanyaan)
