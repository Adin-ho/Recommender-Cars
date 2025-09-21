FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# deps sistem minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY frontend /app/frontend

# Port dari Zeabur -> ${PORT}
ENV PORT=8000
# Default: LLM aktif (panggil Ollama eksternal); bisa dimatikan dengan ENABLE_LLM=0
ENV ENABLE_LLM=1 \
    ALLOWED_ORIGINS=*

CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
