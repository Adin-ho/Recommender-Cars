FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Cache model ke lokasi yg bisa ditulis
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# /data writable untuk cache & chroma
RUN mkdir -p /data && chmod -R 777 /data

# Prefetch model (hindari heredoc supaya aman di Windows)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY frontend ./frontend

ENV PORT=7860 ENABLE_RAG=1 CHROMA_DIR=/data/chroma
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
