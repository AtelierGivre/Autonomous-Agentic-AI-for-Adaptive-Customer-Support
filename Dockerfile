FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production LOG_JSON=true DEBUG_RESPONSES=false \
    DEFAULT_CUSTOMER_ID=
WORKDIR /app

# Run as a non-root user.
RUN useradd --create-home --uid 10001 app

COPY requirements.txt requirements-local.txt ./
# CPU-only torch: the default PyPI wheel bundles CUDA libraries (+2 GB) that a CPU container never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-local.txt

COPY --chown=app:app . .
RUN mkdir -p data storage logs && chown -R app:app data storage logs
USER app

# Bake the embedding model + knowledge index into the image so start-up is fast
# and deterministic, then forbid runtime downloads.
RUN EMBEDDING_BACKEND=hf python -c "from agents_app.rag import KnowledgeBase; KnowledgeBase()"
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).status == 200 else 1)"

# ONE worker on purpose: rate-limit state and SQLite are per-process (see docs/ARCHITECTURE.md, Scaling).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
