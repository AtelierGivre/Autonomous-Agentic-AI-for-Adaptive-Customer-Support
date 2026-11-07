.PHONY: install install-local lint test eval eval-local check offline local run api docker

install:           ## install runtime + dev tools (no torch, no model download)
	pip install -r requirements-dev.txt

install-local:     ## add optional local LLM + Hugging Face embedding dependencies
	pip install -r requirements-local.txt

lint:              ## static checks
	ruff check .

test:              ## 100 offline tests (no keys, no network, no torch)
	pytest -q

eval:              ## golden-dataset quality gate, using whatever LLM_PROVIDER / EMBEDDING_BACKEND is configured
	python -m evals.run_evals --min-pass 0.9

check:             ## what CI runs: lint + tests + offline quality gate
	ruff check . && pytest -q && LLM_PROVIDER=mock EMBEDDING_BACKEND=hash MIN_RETRIEVAL_SCORE=0.1 python -m evals.run_evals --min-pass 0.9

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
