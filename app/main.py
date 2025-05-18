from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.rag_qa import stream_mobil

app = FastAPI()

# Izinkan frontend lokal mengakses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://127.0.0.1:8000"],  # tambahkan 8000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount folder frontend agar bisa akses file statis
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Buka halaman utama saat akses root
@app.get("/")
def root():
    return FileResponse("frontend/index.html")

# Endpoint untuk streaming jawaban
@app.get("/stream")
async def stream(pertanyaan: str):
    return await stream_mobil(pertanyaan)
