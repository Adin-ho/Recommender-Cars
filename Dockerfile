FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# deps sistem minimal (untuk pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY frontend /app/frontend

ENV PORT=8000
ENV ENABLE_RAG=0 ENABLE_LLM=0 ALLOWED_ORIGINS=* CHROMA_PERSIST_DIR=/app/chroma

CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
