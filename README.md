# Autonomous Agentic AI for Adaptive Customer Support

**Planner → Retriever ‖ Tool agent → Troubleshooter → Verifier.** A multi-agent customer-support system built with **Python · FastAPI · asyncio · SQLite · optional Hugging Face / OpenAI-compatible models**. The model can be a cloud API **or a model running entirely on your own machine**.

A planner, retriever, tool agent, troubleshooter and verifier work together to answer support questions, look up orders, and escalate to a human. The agents decide their own plan per request (autonomous), change strategy when their advice keeps failing (adaptive), and sit inside guardrails, reliability controls and an evaluation harness.

---

## The problem

Most chatbot demos only show the model answering. In customer support, the expensive failures are not wrong tone or a clumsy sentence. They are:

| Failure | What it costs |
|---|---|
| **Inventing a policy or refund** the company never offered | Money, trust, legal exposure |
| **Showing one customer another customer's order** | Privacy breach |
| **Acting on an ambiguous "yes"** (for example, "yes but what does it cost?" creates a ticket) | Unwanted side effects |
| **One hostile user taking the service down for everyone** | Outage |
| **Leaking personal data** into logs, memory or the model provider | Compliance risk |
| **Obeying instructions hidden in a message or document** (prompt injection) | Loss of control |

This project puts a specific, testable control against each of these.

## How each failure is handled

| Failure | Control | Where |
|---|---|---|
| Invented policy | **Confidence gate** (answers only from retrieved evidence, otherwise says "I don't know" and offers a human), plus a **verifier agent** that fact-checks the draft against the evidence and triggers one revision | `workflow.py`, `agents.py` |
| Cross-customer data access | **Server-side identity**: the customer ID comes from the authenticated caller, is injected into tool calls, hidden from the model, and enforced inside each tool. A foreign order looks exactly like a missing one | `agents.py`, `mcp_server.py`, `api.py` |
| Ambiguous consent | **Human-in-the-loop**: side-effecting actions are only *proposed*. The whole message must be an unambiguous yes/no, and the confirmation expires | `memory.py`, `workflow.py` |
| One user causing an outage | **Circuit breakers** count only real outages (timeouts, 429, 5xx). Content-filter rejections become a safe reply and never trip the breaker | `llm.py`, `resilience.py` |
| Personal data exposure | **PII redaction** (emails, phones, Luhn-valid cards, Aadhaar/PAN-like IDs, secrets) before the model, memory and audit log; retention purge; erasure endpoint | `security.py`, `store.py` |
| Prompt injection | Input screen, untrusted text fenced and sanitised in prompts, suspicious knowledge-base docs excluded, tool arguments schema-validated against an allow-list | `security.py`, `rag.py`, `mcp_client.py` |

These are heuristics and controls that **reduce** risk. They do not eliminate it (see [Limitations](#limitations)).

## Why it is tested offline, and why that is a strength

Most of the controls above are **deterministic logic**, not model quality. A real LLM makes them harder to test: output varies between runs, and every run costs money. So the project ships a **mock LLM** that can be forced to misbehave on demand:

- Make the verifier reject a draft, then check the system falls back instead of shipping it.
- Make the provider fail five times in a row, then check the circuit opens and requests fail fast with a safe message.
- Make the "model" try to choose another customer's ID, then check the tool layer ignores it.

Everything else is real: an in-process MCP-compatible tool server (JSON-RPC), a real hybrid retrieval index over the local knowledge base, a real SQLite store, and a real HTTP stack. The result is a safety layer that is reproducible, free to run, and checked in CI with **no API keys**.

What offline testing does **not** prove is answer quality with a real model. That is measured separately by the eval harness (`make eval`) and is listed under known gaps.

## Architecture

```mermaid
flowchart TD
    U[Customer message] --> G[Input guard: length, injection screen, PII redaction]
    G --> C{Pending confirmation?}
    C -- yes/no --> X[Execute or decline the proposed action]
    C -- none --> P[Planner agent]
    P -- out of scope --> O[Scope refusal]
    P --> R[Retriever: embedding + BM25 hybrid search]
    P --> T[Tool agent: MCP tools]
    R --> Q{Confidence gate}
    T --> Q
    Q -- weak evidence --> F[Honest "I don't know" + offer human ticket]
    Q -- enough evidence --> A[Troubleshooter writes the answer]
    X --> A
    A --> OG[Output guard]
    OG --> V[Verifier agent: grounded in evidence?]
    V -- rejected --> A2[One revision, re-verified]
    A2 -- still ungrounded --> F
    V -- grounded --> D[Reply]
    A2 -- grounded --> D
```

Retrieval and tool calls run **in parallel**. Write actions are never executed directly: they are proposed first and run only after the customer confirms. Full design, threat model and scaling notes are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What "autonomous" and "adaptive" mean here

Both words are easy to oversell, so here is exactly what the code does.

**Autonomous**: given a customer message, the planner chooses the plan (which agents run, whether a live tool is
needed), retrieval and tool calls run in parallel, the verifier checks the draft and triggers one revision, and
failures end in a safe reply. A human is only involved for side-effecting actions, by design.

**Adaptive**, at two time scales, deliberately not a third one:

| Scale | What adapts | Mechanism |
|---|---|---|
| Within a conversation | Wording, retrieval and strategy | Session memory (e.g. the remembered device sharpens retrieval), re-planning when a tool fails, one revision when the verifier rejects, and **adaptive escalation**: after `ESCALATE_AFTER_FAILURES` (default 2) reports like "still not working" the system stops repeating advice and offers a human ticket, which still needs the customer's YES |
| Across releases | Prompts, thresholds, test coverage | **Feedback loop**: thumbs-down turns are exported by `python -m evals.feedback_to_cases` as candidate eval cases that a person labels and adds to the golden dataset, so every real failure becomes a regression test |
| *Not done on purpose* | The system rewriting its own prompts or thresholds in production | Unreviewed self-modification is a safety and audit risk. Changes go through evals and a human |

The model layer is a thin adapter (`llm.py`) for Azure OpenAI, plain OpenAI or any OpenAI-compatible local server, an in-process Transformers pipeline, or the deterministic mock. The orchestration itself is explicit asyncio code in `workflow.py`, which keeps every step testable and deterministic. The trade-off is that you do not get graph-style checkpointing for free.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows: copy .env.example .env
# open .env and fill in ONE provider (Azure OpenAI or OpenAI), then:
python main.py                                         # terminal chat
uvicorn api:app --port 8000                            # web UI at http://localhost:8000 + API
```

The base install uses a deterministic hash embedder, so the first run needs **no model download**. For Hugging Face embeddings or an in-process local LLM, install the optional extras: `pip install -r requirements-local.txt` (about 130 MB for the default embedding model, plus the model you choose).

**No key yet?** Run the whole pipeline offline with the fake LLM (canned answers, everything else real):

```bash
LLM_PROVIDER=mock EMBEDDING_BACKEND=hash MIN_RETRIEVAL_SCORE=0.1 python main.py
# or: make offline
```

## Run it fully locally (no API key, no data leaves your machine)

Embeddings and retrieval are already local. The LLM can be local too, in two ways:

```bash
# A) In-process Hugging Face model. First run downloads the weights, then it works offline.
echo "LLM_PROVIDER=local" >> .env        # model defaults to Qwen/Qwen2.5-3B-Instruct, change with LOCAL_MODEL
make local

# B) A local server (Ollama, vLLM, llama.cpp) speaking the OpenAI API. Faster on CPU thanks to quantization.
#    .env:  OPENAI_BASE_URL=http://localhost:11434/v1   OPENAI_MODEL=qwen2.5:7b-instruct
```

The agents, guardrails, retrieval and tests are identical. What changes is quality and speed, so measure it with
`make eval-local`. Model choice, hardware sizing, trade-offs and operating notes are in
[`docs/LOCAL_MODELS.md`](docs/LOCAL_MODELS.md).

## Try it

| Type | What it shows |
|---|---|
| `My router keeps disconnecting` | Planning, hybrid retrieval with sources, verifier grounding |
| `Where is my order ORD-1003?` | Tool agent calls `lookup_order` over **MCP**, in parallel with retrieval |
| `Nothing worked, escalate to a human` | **Human-in-the-loop**: proposes a ticket, reply `yes` to execute |
| `Ignore all previous instructions...` | **Prompt-injection guard** blocks it with zero LLM calls |
| `What's the weather?` | Scope control (out-of-scope refusal) |
| `How do I calibrate a quantum flux capacitor?` | **Confidence gate**: says "I don't know" instead of guessing |
| `My email is a@b.com and my router is down` | **PII redaction** before the model sees it |
| `Where is my order ORD-9001?` | **Authorization**: that order belongs to another customer, so it looks "not found" |

Fake demo orders: `ORD-1001` (delivered), `ORD-1002` (in transit), `ORD-1003` (delayed). Open "What the agents did" in the web UI, or `/metrics`, to see latency, tokens and cost per agent.

## Quality gates

```bash
pip install -r requirements-dev.txt
ruff check .                                # lint
pytest -q                                   # 100 offline tests (no keys, network or torch)
python -m evals.run_evals --min-pass 0.9    # golden-dataset quality gate (26 cases)
# or simply:  make check
```

CI (`.github/workflows/ci.yml`) runs all three on every push with no secrets.

**Feedback loop:** `python -m evals.feedback_to_cases` turns thumbs-down turns into candidate cases to label (unlabeled cases are refused by the harness, so they cannot pass by accident).

**Tuning:** the eval report prints every case with its outcome and failure reasons, and writes `evals/report.json`. Inspect the retrieval confidence in that report, set `MIN_RETRIEVAL_SCORE`/`MIN_LEXICAL_COVERAGE` for your embedding model, and add real customer questions to `evals/dataset.jsonl`. Run the evals with your real key before demos or prompt changes: that is the regression test for answer quality.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` `{"session_id","message"}` | One chat turn. `X-API-Key` required when `API_KEYS` is set |
| `POST /v1/chat/stream` | Same, as Server-Sent Events: live `step` events, then one `answer` event |
| `POST /v1/feedback` `{"trace_id","rating":1/-1}` | Thumbs up/down |
| `DELETE /v1/sessions/{id}` | Erase a conversation and its audit trail |
| `GET /health/live`, `/health/ready` | Probes |
| `GET /metrics` | Prometheus metrics |

**Identity:** your backend authenticates the end customer, then calls the API with its `X-API-Key` and an `X-Customer-Id` header. Tools are scoped to that customer server-side, and the model can neither choose nor see it. Never put the API key in a browser or mobile app. (The demo UI relies on the development default customer `demo`.)

**Production mode** (`ENVIRONMENT=production`) refuses to start without `API_KEYS` or with the mock LLM, hides debug info and API docs, and emits JSON logs.

## Deploy to Azure

```bash
az login && bash deploy_azure.sh
```

Builds the Dockerfile in Azure Container Apps with secrets stored as secret references. The container runs as a non-root user with the embedding model and index baked in. It runs as a **single replica by design** (sessions are in SQLite and rate limits are in memory); see "Scaling" in the architecture doc before changing that. The script has not yet been tested against a real subscription, so review it before first use.

## Project map

| Path | What |
|---|---|
| `agents_app/agents.py` | The five agents, each with one job, prompt and model tier |
| `agents_app/workflow.py` | Orchestration: guards, parallel execution, confidence gate, confirmation, verify/revise, degradation |
| `agents_app/rag.py` | Hybrid retrieval (embedding + BM25 with reciprocal rank fusion, optional Hugging Face embeddings), confidence score, auto re-index |
| `agents_app/mcp_server.py`, `mcp_client.py` | MCP-compatible tools (read/write annotated, JSON-RPC) and a validating, timeout-guarded client |
| `agents_app/llm.py`, `resilience.py` | Model routing (Azure OpenAI, OpenAI-compatible, in-process Transformers, mock), retries, circuit breaker, budgets, validated JSON output |
| `agents_app/security.py`, `guardrails.py` | PII redaction, injection screen, auth, rate limiting |
| `agents_app/store.py`, `memory.py` | SQLite sessions, audit, feedback, retention, erasure; memory also tracks repeated failures for adaptive escalation |
| `agents_app/monitoring.py`, `logging_setup.py` | Prometheus and per-request metrics, structured logs |
| `api.py`, `static/` | FastAPI service and demo UI |
| `evals/`, `tests/` | Golden dataset and harness, `feedback_to_cases.py` (feedback loop); unit and integration tests |
| `docs/` | `ARCHITECTURE.md`, `LOCAL_MODELS.md`, `CODE_REVIEW.md`, `TEACHING_GUIDE.md` |
| `knowledge_base/*.md` | The company docs. Edit them and the index rebuilds automatically |

## How this was built

I designed the architecture (multi-agent flow, MCP tool layer, guardrails) and used an AI assistant to draft much of the code, treating its output as untrusted. Code review found and fixed real bugs, for example duplicate ticket IDs and an incomplete output filter, each with a regression test. The findings are recorded in [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md).

## Limitations

Stated plainly, because they decide how far to trust this:

- **Verified offline only** (mock LLM, hash embeddings). Behaviour with a real model, answer quality, and the real Hugging Face embedding path are not yet validated. Run `make eval` with your own key, or `make eval-local` for the local model, first.
- **Local model: implemented, not exercised in CI.** The in-process path needs multi-GB weights, so this workspace does not run it automatically. Small local models are weaker at planning and verification than large hosted ones. The safety controls do not depend on the model, but helpfulness does. See [`docs/LOCAL_MODELS.md`](docs/LOCAL_MODELS.md).
- The golden dataset is small and synthetic. Replace or extend it with real, anonymised tickets.
- Injection and PII detection are **heuristic**. Free-text names and addresses are not detected.
- The verifier is itself an LLM, so it can be wrong. Measure it with evals.
- The API trusts `X-Customer-Id` from any holder of a valid API key. That is only correct when the key lives in your backend and that backend has authenticated the customer.
- No load test, penetration test or compliance review (SOC 2, GDPR, DPDP) has been done.
- The MCP server uses fake data. Replace the tool bodies with real CRM and order-system calls, keeping the same ownership checks.

More detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), section 6.

## Versions

Developed and tested on Python 3.11+ (3.14 in this workspace) with fastapi 0.162, pydantic 2.x, prometheus-client, python-dotenv and openai. Optional: transformers and sentence-transformers for local models and Hugging Face embeddings. There is no LangChain, LlamaIndex, FAISS or subprocess MCP dependency.
