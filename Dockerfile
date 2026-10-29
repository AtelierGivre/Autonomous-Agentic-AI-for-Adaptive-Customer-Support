FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pre-download the Hugging Face embedding model and pre-build the FAISS index at build time,
# so the container starts fast.
RUN python -c "from agents_app.rag import KnowledgeBase; KnowledgeBase()"
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
