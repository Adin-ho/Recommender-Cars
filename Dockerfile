# Gunakan image multi-arch (bisa amd64/arm64)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# dependensi untuk build beberapa wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY app ./app
COPY frontend ./frontend

# default env (Zeabur akan override)
ENV PORT=8000 \
    ENABLE_RAG=1 \
    CHROMA_PERSIST_DIR=/data/chroma

EXPOSE 8000
# Jalankan uvicorn. Gunakan PORT dari env Zeabur.
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
