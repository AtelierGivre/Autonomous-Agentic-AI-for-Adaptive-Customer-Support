# Running with local models

Everything in the request path can run on one machine. No API key is required,
and customer text never leaves the process unless you choose a hosted provider.

Two independent choices:

1. **Embeddings** — `EMBEDDING_BACKEND=hash` (default, no download) or `hf`
   (sentence-transformers).
2. **The LLM** — `mock` (offline demo), `local` (in-process Transformers), or
   `openai` pointed at a local OpenAI-compatible server.

The safety controls, retrieval pipeline, tools and tests are identical in every
mode. What changes is answer quality, latency and memory use.

## 1. In-process Hugging Face model

Install the optional extras:

```bash
pip install -r requirements-local.txt
```

`transformers` is the runtime; `sentence-transformers` brings torch and the
embedding model. For a CPU-only machine, install the CPU torch wheel first so
pip does not pull ~2 GB of CUDA libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-local.txt
```

Then configure `.env` (or the environment):

```dotenv
LLM_PROVIDER=local
LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct
# Optional smaller model for the planner, tool agent and verifier.
LOCAL_MODEL_FAST=
# Reproducible deployments: pin a commit in the model repo.
LOCAL_MODEL_REVISION=
# auto | cpu | cuda | mps
LOCAL_DEVICE=auto
# auto | bfloat16 | float16 | float32
LOCAL_DTYPE=auto
LOCAL_MAX_NEW_TOKENS=400
LOCAL_TEMPERATURE=0.2
```

Run the terminal client or the API:

```bash
python main.py
# or
uvicorn api:app --port 8000
```

### What each setting does

| Setting | Effect in `agents_app/llm.py` |
|---|---|
| `LOCAL_MODEL` | Model passed to `transformers.pipeline("text-generation", ...)`. |
| `LOCAL_MODEL_FAST` | Used only for `fast=True` calls: planner, verifier and JSON-style agents. Falls back to `LOCAL_MODEL`. |
| `LOCAL_MODEL_REVISION` | Passed as `revision=` so a rebuild uses the same weights. Empty means the default branch. |
| `LOCAL_DEVICE` | `auto` uses `device_map="auto"` (needs `accelerate`). `cpu`, `cuda` and `mps` are passed as `device=`. If the device map is unsupported, the loader retries without it. |
| `LOCAL_DTYPE` | Mapped to `torch.bfloat16`, `torch.float16` or `torch.float32`; `auto` lets Transformers choose. |
| `LOCAL_MAX_NEW_TOKENS` | Generation cap per answer. |
| `LOCAL_TEMPERATURE` | Sampling temperature. The planner and verifier call with `temperature=0` (greedy). |

Pipelines are cached per model name, so using a distinct fast model loads two
models. Leaving `LOCAL_MODEL_FAST` empty shares one copy in memory.

The first run downloads weights from the Hugging Face Hub (several GB). After
that the cache is reused. In a container, download during the build and set
`HF_HUB_OFFLINE=1` at runtime; the Dockerfile shows the pattern.

## 2. A local OpenAI-compatible server

For CPU-friendly quantisation, a server is often faster than an in-process
model. Ollama, vLLM, llama.cpp's server and LM Studio all expose the OpenAI
chat-completions API.

Example with Ollama:

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

`.env`:

```dotenv
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b-instruct
OPENAI_API_KEY=
```

`OPENAI_API_KEY` may stay empty for a server that does not check it. The code
sends the placeholder `local` only when no key is configured, so the OpenAI SDK
is satisfied without a real secret.

This path keeps the application process small: the model lives in the server.
It also lets you switch models without reinstalling dependencies.

## 3. Embeddings

| Backend | When | Trade-off |
|---|---|---|
| `hash` | Default, tests, CI, a zero-download demo | Deterministic, stdlib-only, **lexical** rather than semantic. Good enough for these seven documents; not a general embedding model. |
| `hf` | Real semantic retrieval | Requires `sentence-transformers` + torch; first run downloads `HF_EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`, about 130 MB). |

```dotenv
EMBEDDING_BACKEND=hf
HF_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

**Retune the gate when you change backends.** The two score distributions are
not comparable. The code default is 0.45 (suitable for `hf`); `.env.example`
starts at 0.1 so the zero-download hash path answers on first run. The hash path
also lowers the floor to 0.10 when `MIN_RETRIEVAL_SCORE` is unset. Use the eval
report to pick both thresholds (see section 5).

If `EMBEDDING_BACKEND=hf` is requested but the package or weights are missing,
`KnowledgeBase` falls back to `hash` and reports that at `/health/ready`. It does
not crash.

## 4. Hardware sizing

Approximate, for a single request at a time. These are starting points, not
benchmarks measured in this repo. Model size dominates everything else.

| Model class | Weights (bf16) | Practical RAM/VRAM | Notes |
|---|---|---|---|
| 1.5B–3B | 3–6 GB | 8 GB RAM or 6 GB VRAM | Default `Qwen2.5-3B-Instruct`; usable on a laptop CPU, slower. |
| 7B–8B | 14–16 GB | 16 GB RAM or 8–12 GB VRAM with 4-bit | Better planning/verification; quantised server is the practical CPU option. |
| 13B–14B | 26–28 GB | 32 GB RAM or 16–24 GB VRAM | Diminishing returns for this narrow support task. |

A 4-bit quantised 7B model served by Ollama generally needs 5–7 GB of memory.
CPU throughput is the limiting factor, not disk.

## 5. Measuring quality

The offline gate proves the plumbing, not the model. Measure every local model
you intend to use:

```bash
# Uses whatever LLM_PROVIDER / EMBEDDING_BACKEND / model is configured.
make eval-local

# or directly, with a lower bar appropriate for a small local model:
LLM_PROVIDER=local python -m evals.run_evals --min-pass 0.8
```

Read `evals/report.json`: each case records its outcome, pass/fail and the
failure reason. Common patterns:

- Planner emits invalid JSON → the rule-based fallback still answers, but some
  cases will route differently.
- Verifier rejects correct answers → the system falls back to `no_answer`. That
  is safe but unhelpful; lower `MIN_RETRIEVAL_SCORE` only after checking the
  retrieval confidence in the report.
- Answers drift from the evidence → use a larger instruct model or a tighter
  prompt. Do **not** relax the verifier to make numbers look better.

The repo does not claim speed or quality numbers for any local model. If you
publish numbers, record the model, revision, quantisation, hardware and dataset.

## 6. Operating notes

- **Pin versions.** `LOCAL_MODEL_REVISION` plus a pinned `transformers` version
  makes a deployment reproducible.
- **Check the licence.** Model weights carry their own licence, which may differ
  from this repository's MIT licence. Check it before commercial use.
- **Memory.** Two different models means two copies. A 3B fast model plus a 7B
  answer model can exceed a small VM; measure before deploying.
- **Startup.** Lazy loading means the first request is slow. For a service,
  warm the pipeline at startup or bake the weights into the image.
- **Offline containers.** Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  after the weights are cached, so a network failure cannot change the image's
  behaviour.
- **Logging.** The application logs events and metrics, not prompt bodies. A
  local provider keeps prompts in-process; a hosted provider does not.
- **Production guard.** `ENVIRONMENT=production` refuses to start with
  `LLM_PROVIDER=mock`. Use a real model, local or hosted, for a production run.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: transformers` | Local extras not installed | `pip install -r requirements-local.txt` |
| Out-of-memory during load | Model too large for the device, or two models loaded | Use a smaller model, quantisation, or leave `LOCAL_MODEL_FAST` empty |
| Very slow first response | Weights are downloading or the model is loading lazily | Pre-download/warm the model; use a server with quantisation |
| Planner returns invalid JSON | Small model struggles with strict output | Keep the rule fallback; try a larger instruct model; leave `JSON_MODE=true` |
| `no_answer` for obvious questions | Retrieval gate too strict for this model's embeddings | Inspect `evals/report.json` and `retrieval` debug, then tune thresholds |
| `degraded` outcome | Provider failed, retries exhausted, or breaker open | Check `/health/ready`, provider logs, and `MOCK_LLM_MODE`; wait for the breaker cool-down |
| `EMBEDDING_BACKEND=hf` reports `hash` | Package or weights missing | Install extras / allow the first download; check `/health/ready` |
