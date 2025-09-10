# Image multi-arch (amd64/arm64)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Tools yang sering dibutuhkan wheel compiled
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 1) Torch CPU wheel dari repo resmi (hindari compile)
RUN pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# 2) Dependensi lain
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 3) Copy source
COPY app ./app
COPY frontend ./frontend

# 4) (Opsional) Pre-download model embedding agar start cepat
#    Tidak wajib; bisa kamu comment kalau build terlalu lama.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
PY

# Default env (Zeabur override via Variables)
ENV PORT=8000 \
    ENABLE_RAG=1 \
    BUILD_EMBED_ON_START=0 \
    CHROMA_PERSIST_DIR=/data/chroma

EXPOSE 8000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
