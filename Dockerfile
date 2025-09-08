FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# === cache model di folder yang bisa ditulis ===
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# siapkan folder cache yang writable
RUN mkdir -p $HF_HOME && chmod -R 777 /data

# (opsional) prefetch model agar startup cepat
RUN python - <<'PY'\nfrom sentence_transformers import SentenceTransformer\nSentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\nPY

COPY app ./app
COPY frontend ./frontend

ENV PORT=7860 ENABLE_RAG=1
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
