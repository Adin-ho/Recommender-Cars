import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="ChatCars")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Static frontend
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Routers
from app.rule_based import router as rule_router
app.include_router(rule_router)

# Kalau tetap ingin RAG, set ENABLE_RAG=1
if os.getenv("ENABLE_RAG", "0") == "1":
    from app.rag_qa import router as rag_router
    app.include_router(rag_router)
