# Dockerfile – fix 502 (tak set PORT default; gunakan PORT dari Zeabur)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pasang Torch CPU (tanpa compile)
RUN pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY app ./app
COPY frontend ./frontend

# (Opsional) kalau build sering timeout, COMMENT baris di bawah ini
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ⚠️ JANGAN set ENV PORT di sini—biarkan Zeabur yang isi
ENV ENABLE_RAG=1 \
    BUILD_EMBED_ON_START=0 \
    CHROMA_PERSIST_DIR=/data/chroma

# `PORT` akan disuntik oleh Zeabur saat runtime
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
