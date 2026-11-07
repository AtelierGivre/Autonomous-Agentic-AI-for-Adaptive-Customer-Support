.PHONY: install install-local install-rag lint test eval eval-retrieval bench-rag eval-local check check-rag offline local run api docker

install:           ## install runtime + dev tools (no torch, no model download)
	pip install -r requirements-dev.txt

install-local:     ## add optional local LLM + Hugging Face embedding dependencies
	pip install -r requirements-local.txt

install-rag:       ## add optional FAISS vector index + LlamaIndex chunking
	pip install -r requirements-rag.txt

lint:              ## static checks
	ruff check .

test:              ## 116 offline tests (no keys, no network, no torch)
	pytest -q

eval:              ## golden-dataset quality gate, using whatever LLM_PROVIDER / EMBEDDING_BACKEND is configured
	python -m evals.run_evals --min-pass 0.9

eval-retrieval:    ## labelled retrieval gate (recall@k, MRR) for the configured backends
	python -m evals.run_retrieval --min-recall 0.8

bench-rag:         ## vector-store build/query benchmark (builtin vs FAISS)
	python -m evals.benchmark_rag --backend faiss --sizes 1000,10000

check:             ## what CI runs: lint + tests + offline quality + retrieval gates
	ruff check . && pytest -q && LLM_PROVIDER=mock EMBEDDING_BACKEND=hash MIN_RETRIEVAL_SCORE=0.1 python -m evals.run_evals --min-pass 0.9 && python -m evals.run_retrieval --min-recall 0.8

check-rag:         ## exercise the optional FAISS + LlamaIndex backends
	VECTOR_STORE=faiss INGEST_BACKEND=llamaindex pytest -q && VECTOR_STORE=faiss INGEST_BACKEND=llamaindex python -m evals.run_retrieval --min-recall 0.8

local:             ## chat in the terminal with a Hugging Face model running on this machine (no API key)
	LLM_PROVIDER=local python main.py

eval-local:        ## golden-dataset quality gate using the local model (this is how you measure its quality)
	LLM_PROVIDER=local python -m evals.run_evals --min-pass 0.8

offline:           ## chat in the terminal with a fake LLM (no API key)
	LLM_PROVIDER=mock EMBEDDING_BACKEND=hash MIN_RETRIEVAL_SCORE=0.1 python main.py

run:               ## chat in the terminal (needs .env filled in)
	python main.py

api:               ## web UI + API on http://localhost:8000
	uvicorn api:app --reload --port 8000

docker:            ## build the production image
	docker build -t agentic-support .
