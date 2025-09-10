# Dockerfile (fixed: no heredoc)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Tooling dasar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Torch CPU wheels (tanpa compile)
RUN pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Dependensi lain
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Source
COPY app ./app
COPY frontend ./frontend

# (Opsional) pre-download model embedding – gunakan one-liner, bukan heredoc.
# Kalau build masih lama, COMMENT baris ini.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Env default (Zeabur override di Variables)
ENV PORT=8000 \
    ENABLE_RAG=1 \
    BUILD_EMBED_ON_START=0 \
    CHROMA_PERSIST_DIR=/data/chroma

EXPOSE 8000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
