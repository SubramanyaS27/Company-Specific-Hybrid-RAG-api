FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=10000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt fastapi uvicorn tokenizers

COPY rag_pipeline.py api.py chunks.json bm25_index.pkl ./
COPY qdrant_data ./qdrant_data

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]