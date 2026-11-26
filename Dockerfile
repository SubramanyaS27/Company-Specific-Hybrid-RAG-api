FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface \
    PORT=10000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY rag_pipeline.py api.py chunks.json bm25_index.pkl ./
COPY qdrant_data ./qdrant_data

RUN python -c "from tokenizers import Tokenizer; from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; Tokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2'); TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]