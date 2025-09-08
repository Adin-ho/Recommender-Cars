FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Cache model di lokasi yang bisa ditulis oleh Space
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pastikan folder cache ada & writable
RUN mkdir -p $HF_HOME && chmod -R 777 /data

# Prefetch model (hindari heredoc supaya tidak kena CRLF)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY frontend ./frontend

ENV PORT=7860 ENABLE_RAG=1
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
