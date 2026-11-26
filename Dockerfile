FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=10000

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY rag_pipeline.py api.py chunks.json bm25_index.pkl ./
COPY qdrant_data ./qdrant_data

RUN python -c "from transformers import AutoTokenizer; from sentence_transformers import SentenceTransformer, CrossEncoder; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]