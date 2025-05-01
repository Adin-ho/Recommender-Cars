from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.recommender import recommend
from app.parser import extract_preferences
from pydantic import BaseModel


app = FastAPI()

class ChatRequest(BaseModel):
    message: str

# Izinkan request dari semua domain (frontend lokal)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
async def chat(req: ChatRequest):
    user_input = req.message
    prefs = extract_preferences(user_input)
    results = recommend(prefs)

    response = []
    for _, row in results.iterrows():
        response.append({
            "nama_mobil": row["Nama Mobil"],
            "tahun": row["Tahun"],
            "harga": f"Rp{row['Harga']:,}",
            "transmisi": row["Transmisi"],
            "bahan_bakar": row["Bahan Bakar"]
        })

    return {"reply": response}
