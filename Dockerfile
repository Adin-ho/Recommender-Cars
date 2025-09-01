FROM python:3.11-slim

# Paket sistem minimum agar pip build lancar
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Env: RAG CPU + cache model + port Spaces
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface \
    ENABLE_RAG=1 \
    USE_OLLAMA=0 \
    PORT=7860

# Salin source code
COPY . .

# Pre-build index Chroma saat build (biar startup cepat)
RUN python -m app.embedding

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
