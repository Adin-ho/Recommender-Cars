FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Tambah alat build minimal yang sering dibutuhkan chromadb/hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ cmake git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# (opsional) percepat pip
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY frontend /app/frontend

ENV PORT=8000 \
    ENABLE_RAG=1 \
    BUILD_EMBED_ON_START=0 \
    ENABLE_LLM=0 \
    ALLOWED_ORIGINS=* \
    CHROMA_PERSIST_DIR=/app/chroma

CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
