# Architecture

This document describes what the code actually does. Where a claim in the README
is a design goal rather than a measured result, it says so.

- Runtime: Python 3.11+ (`asyncio`), FastAPI, SQLite, `pydantic` v2.
- Orchestration: explicit code in `agents_app/workflow.py`. No LangChain,
  LangGraph or subprocess MCP dependency is required, and the workflow is not
  built on any framework. FAISS and LlamaIndex are **optional, opt-in** retrieval
  backends (`requirements-rag.txt`); the default is dependency-free.
- Offline default: deterministic mock LLM + hash embeddings. Optional local
  models are described in [`LOCAL_MODELS.md`](LOCAL_MODELS.md).

## 1. System overview

```text
Browser / backend client
        |
        |  POST /v1/chat, /v1/chat/stream, /v1/feedback
        v
FastAPI (api.py)
  - API-key auth, per-key + per-customer rate limits, 64 KB body limit
  - SQLite load/save of one conversation, one lock per session
        |
        v
SupportWorkflow (agents_app/workflow.py)
  - input screen -> PII redaction -> pending confirmation
  - planner (JSON, rule fallback)
  - retrieval || tool call
  - confidence gate
  - troubleshooter -> verifier -> one revision
  - output guard
        |
        +--> KnowledgeBase (agents_app/rag.py): local markdown, hybrid BM25 + vectors
        +--> MCPClient -> ToolServer (agents_app/mcp_server.py): orders and tickets
        +--> LLM (agents_app/llm.py): Azure / OpenAI-compatible / Transformers / mock
```

The UI in `static/` talks to `/v1/chat/stream`, renders every string with
`textContent`, and never holds an API key. It relies on the development default
customer (`demo`). In a real deployment the browser calls your own backend,
which authenticates the customer and then asserts `X-Customer-Id` server-side.

## 2. Request lifecycle

`SupportWorkflow.handle()` wraps `_handle()` in `asyncio.wait_for` using
`REQUEST_TIMEOUT_S` (default 60 s). A timeout returns a safe reply with outcome
`timeout`; it never raises to the caller.

The order inside `_handle()` is deliberate:

| # | Step | Code | Behaviour |
|---|---|---|---|
| 1 | Trace registration | `store.record_trace()` | A `trace_id` is created before any work so feedback can reference it. |
| 2 | Input screen | `screen_input()` | Length > 4000 chars or a known injection pattern is blocked **before any model call**. The raw text is not stored; memory receives `[blocked input]`. |
| 3 | PII redaction | `redact_pii()` | Emails, phones, Luhn-valid cards, Aadhaar/PAN-like IDs and secrets become `[EMAIL]`, `[CARD]`, … before the model, memory or audit trail sees them. |
| 4 | Pending confirmation | `ConversationMemory.pending` | An unexpired proposal consumes the turn. Exact whole-message yes/no only. A reply that starts with yes/no but contains more is ambiguous: ask again, never execute. Any other message cancels the proposal with no side effect and is handled normally. |
| 5 | Failure counter | `_FAILURE_RE` | Messages such as “still not working” increment `failure_count`; it drives adaptive escalation. |
| 6 | Planner | `Planner.plan()` | Returns intent, whether retrieval/tools are needed, and an optional proposed action. JSON is validated against a schema; a rule-based plan is always the fallback. |
| 7 | Scope control | `plan.out_of_scope` | Off-topic requests are refused without retrieval or model drafting. |
| 8 | Write proposal | `_propose()` | Creating a ticket is a side effect, so it is only proposed. It executes after a clear YES. `CONFIRM_WRITE_ACTIONS=false` is an explicit opt-out. |
| 9 | Parallel work | `asyncio.gather()` | Retrieval and the tool call run concurrently. Retrieval is skipped when the plan does not need it. |
| 10 | Confidence gate | `KnowledgeBase.is_confident()` | Both the best fused score and lexical coverage must pass. A successful tool result bypasses the document gate. Otherwise the reply is an honest “I don't know” plus a ticket offer (still needs YES). |
| 11 | Draft | `Troubleshooter.write()` | Real providers receive fenced (untrusted) evidence and a strict “do not invent” system prompt. Offline mode assembles a deterministic answer from the best matching sentences. |
| 12 | Verify and revise | `Verifier.verify()` | The draft is checked against evidence/tool output. One revision is allowed. A second rejection becomes `no_answer`, never a shipped guess. |
| 13 | Output guard | `output_guard()` | Strips internal fence markers, caps length, then memory and metrics are updated. |

Failure paths are explicit outcomes rather than exceptions: `blocked`,
`out_of_scope`, `proposed`, `awaiting_confirmation`, `escalated`, `declined`,
`no_answer`, `budget`, `timeout`, `degraded` and `answered`.

Provider failures (including an open circuit breaker and content-filter
rejections) become `degraded` with a safe message. They never turn into a raw
500 and never create a ticket.

## 3. Agents and contracts

| Agent | Job | Real-provider contract | Offline behaviour |
|---|---|---|---|
| **Planner** | Choose the smallest plan that can answer | JSON: `intent`, `need_retrieval`, `need_tools`, `tool`, `order_id` | Deterministic rules (order ID → tool, explicit human request → escalation, failure threshold → escalation, known off-topic words → out of scope). |
| **Retriever** | Fetch evidence | Hybrid search over the local index | Same search; embeddings are hash-based if `sentence-transformers` is absent. |
| **Tool agent** | Call one allow-listed tool | `lookup_order`, `list_orders`, `create_ticket`, `escalate_to_human` | In-process `ToolServer` with fake orders. |
| **Troubleshooter** | Write the customer reply | Answer only from supplied evidence; ≤150 words | Sentence-level extract from the top relevant chunk, or a tool-based status summary. |
| **Verifier** | Decide whether the draft is grounded | JSON: `grounded`, `reason` | Token-overlap check against evidence/tool output; refuses drafts with no support. |

The planner, verifier and tool arguments are JSON. `llm.parse_json_object()`
tolerates code fences and surrounding prose, but a missing/invalid object falls
back to the safe path rather than being trusted.

The five agents share one `RequestBudget` per turn: `MAX_LLM_CALLS_PER_REQUEST`
(default 10) and `MAX_COST_PER_REQUEST_USD` (default $0.05). Exceeding either
returns the `budget` outcome.

## 4. Retrieval

`agents_app/rag.py` is a small, inspectable hybrid index.

**Chunking.** Knowledge-base files are split on blank lines into paragraphs and
packed up to `CHUNK_SIZE` (default 1800 characters) with `CHUNK_OVERLAP`
(default 200 characters). The overlap starts on a word boundary. The shipped
seven documents currently produce seven chunks, so `RETRIEVAL_TOP_K=4` often
returns most of the corpus; this is a demo-scale index, not a benchmark.

`INGEST_BACKEND` selects the chunker:

- `builtin` (default): the paragraph packer above, no dependency.
- `llamaindex`: `SentenceSplitter` from the optional LlamaIndex package,
  calibrated with a character tokenizer and a local sentence splitter so
  `CHUNK_SIZE` / `CHUNK_OVERLAP` keep the same meaning. It does not download
  tokenizer data.
- `auto`: LlamaIndex when importable, otherwise builtin.

Both return plain strings; embedding, tokenisation and scoring stay in
`rag.py`. A failed optional import degrades to builtin rather than breaking
retrieval.

**Vector index.** `VECTOR_STORE` selects where the normalised vectors are
searched:

- `builtin` (default): exact dot-product scan over Python lists, no dependency.
- `faiss`: optional FAISS `IndexFlatIP`, exact for these normalised vectors.
  The index and a configuration fingerprint are cached under `INDEX_DIR`
  (`index.faiss`, `index.meta.json`). A changed document, embedding backend or
  chunk setting invalidates the cache and rebuilds.
- `auto`: FAISS when importable, otherwise builtin.

`IndexFlatIP` is exact, so ranking parity with the built-in scan is asserted in
`tests/test_vector_store.py`. The measured gain is query latency at scale
(about 459 ms to 1.2 ms p50 at 10,000 synthetic 256-dim vectors on the
development machine); at seven chunks the two are equivalent. Persistence saves
rebuild time but not model loading.

**Embeddings.** Two backends:

- `hash` (default when `sentence-transformers` is unavailable): a 256-dimension
  signed feature hash of unigrams and bigrams, L2-normalised. It is
  deterministic and dependency-free, but it is **lexical, not semantic**.
- `hf`: `sentence-transformers` with `HF_EMBEDDING_MODEL` (default
  `BAAI/bge-small-en-v1.5`), normalised vectors. Install
  `requirements-local.txt` to enable it.

**Lexical search.** BM25 with `k1=1.5`, `b=0.75`, normalised by the best score
in the corpus. Text is lowercased, stop-worded and lightly stemmed
(`disconnecting` → `disconnect`). A tiny thesaurus expands query-only terms
such as “twice”/“double” → “duplicate” and “parcel”/“package” → “order,
delivery”.

**Fusion and scoring.** Vector and BM25 rankings are fused with reciprocal rank
fusion (weight 0.65 / 0.35, `k=60`). The returned score blends:

```text
0.45 * cosine
+ 0.20 * normalised BM25
+ 0.15 * normalised RRF
+ 0.20 * query-token coverage
```

`coverage` is the fraction of expanded query tokens present in the chunk.
The planner's intent adds a small bonus to the document that normally owns the
topic: billing → `billing.md` (+0.20), account → `account.md` (+0.20),
troubleshooting → router/warranty/shipping (+0.08), order status →
`shipping.md` (+0.08). This is a bonus, not a filter, so unusual questions can
still be answered from any document.

**Confidence gate.** `is_confident()` requires the top result to satisfy both
`MIN_RETRIEVAL_SCORE` and `MIN_LEXICAL_COVERAGE`. The two scales differ by
backend: the code default is 0.45 (suitable for Hugging Face embeddings), while
`.env.example` starts at 0.1 so a base install answers on first run. The hash
path also lowers the floor to 0.10 when `MIN_RETRIEVAL_SCORE` is unset. Set both
values from the eval report, not from intuition.

**Freshness.** `refresh_if_changed()` compares each file's path, mtime and size;
editing a knowledge-base file rebuilds the index on the next search. Documents
that look like prompt injection are skipped when `KB_SKIP_SUSPICIOUS=true`.

## 5. Tool layer (MCP-compatible)

`ToolServer` exposes `ping`, `tools/list` and `tools/call` over an in-process
surface, and can also speak newline-delimited JSON-RPC on stdio:

```bash
python -m agents_app.mcp_server
```

The default client does **not** spawn a subprocess. That is a deliberate
trade-off: offline tests start in milliseconds and there is no extra process to
supervise. The JSON-RPC shape keeps the door open for a real MCP transport.

`MCPClient` enforces the security boundary before a tool runs:

- allow-listed tool names only;
- allow-listed argument names per tool;
- required arguments are checked;
- a 5-second timeout per call;
- the customer ID is passed **separately** from model-supplied arguments, so the
  model can neither choose nor see it.

`lookup_order` returns the same “not found” shape for a missing order and for an
order owned by a different customer. `create_ticket` is idempotent per
(customer, subject), protected by a lock, so a retry or double-confirm cannot
allocate the same ticket sequence twice.

## 6. Threat model and limitations

| Asset | Threat | Control in this repo | Residual risk |
|---|---|---|---|
| Company policy | Model invents a refund/date | Confidence gate, evidence-only prompt, verifier + one revision, `no_answer` fallback | The verifier is an LLM and can be wrong. Evals measure it; they do not prove it. |
| Another customer's order | Cross-tenant lookup / prompt-injected ID | Customer ID comes from the authenticated caller, is hidden from the model, and is checked inside each tool. Foreign and missing orders are indistinguishable. | The API trusts `X-Customer-Id` from any holder of a valid API key. That is only safe when the key lives in your authenticated backend. |
| Customer PII | Leaks into prompts, memory, logs, feedback | Redaction before model/memory; feedback comments are redacted again; audit rows store IDs and events, not message bodies; retention purge and erasure endpoint | Detection is regex/Luhn-based. Free-text names and postal addresses are **not** detected. Provider-side retention is outside this repo's control. |
| Side effects | Ticket created on an ambiguous “yes” | Whole-message yes/no parser, TTL, ambiguous replies re-ask, unrelated messages cancel, tickets idempotent | A user can still deliberately confirm. The control prevents accidents, not intent. |
| Service availability | Hostile input, provider outage, one noisy customer | 64 KB body limit, 4000-char input, per-key and per-customer sliding windows, request timeout, call/cost budgets, circuit breaker, safe degradation | Rate limits are in memory and the deployment is single-replica. A real multi-replica deployment needs a shared limiter. |
| Prompt injection | Instructions hidden in a message or document | Input screen (zero model calls), fenced untrusted text, control-character stripping, suspicious KB docs skipped, tool allow-list | Heuristics miss paraphrases and indirect injection in ordinary text. |
| Auditability | “What did the agent do?” | `trace_id` per turn, per-turn debug trace, structured logs, Prometheus counters | Debug detail is development-only; production exposes stage names only. |

Not done, on purpose: the system does not rewrite its own prompts or thresholds
in production. Changes go through evals and a human review.

## 7. Reliability and failure handling

- **Retries:** one retry after a failed provider call (`llm.py`).
- **Circuit breaker:** opens after 5 consecutive real outages. Only timeouts,
  429 and 5xx count. A content-filter rejection does not trip it. After a
  cool-down a single half-open probe is allowed.
- **Budgets:** calls and estimated cost are capped per turn.
- **Timeouts:** per LLM call, per tool call, per request.
- **Degradation:** `timeout`, `budget` and `degraded` all return a safe,
  non-committal reply and still record metrics.
- **Verification:** an ungrounded draft is revised once, then replaced with an
  honest `no_answer`; it is never shipped.
- **HTTP boundary:** oversized bodies are rejected before parsing, unexpected
  exceptions become `{"error":"internal_error"}` with a request ID, and security
  headers are set on every response.

## 8. Data, privacy and retention

SQLite (`data/support.db` by default) has four tables:

| Table | Contents |
|---|---|
| `sessions` | JSON memory per session key (turns already redacted) |
| `traces` | `trace_id` → session key + timestamp |
| `audit` | event rows with IDs and timestamps, no message body |
| `feedback` | one rating and optional redacted comment per trace |

The session key is namespaced as `api-key-fingerprint:customer:session_id`, so
one customer cannot read another customer's history even if both guess a session
ID. `DELETE /v1/sessions/{id}` removes the session, its traces, audit rows and
feedback. A background task purges rows older than `RETENTION_DAYS` (default 30)
once an hour.

Writes use one connection guarded by a lock and are offloaded with
`asyncio.to_thread`, so a slow disk does not block the event loop. That design is
sufficient for the documented single replica; it is not a multi-writer design.

## 9. Observability

- `GET /health/live`: process + tool server. Failure returns 503 so the platform
  can restart a stuck single replica.
- `GET /health/ready`: provider, index stats (documents, chunks, embedding
  backend, ingest backend, vector store, thresholds) and tool readiness.
- `GET /metrics`: hand-written Prometheus exposition for turns, LLM calls,
  tokens, estimated cost, latency, outcomes, tool calls and feedback.
- Every turn carries a `trace_id`, an outcome and a debug trace. `DEBUG_RESPONSES`
  is development-only; production hides internals and disables `/docs`.
- `LOG_JSON=true` switches structured logs to one JSON object per line.

## 10. Scaling

The shipped configuration is a **single replica by design**:

- sessions live in local SQLite;
- rate limits live in process memory;
- the knowledge index is built in process;
- the demo tool server holds tickets in memory.

To scale horizontally, change these in order:

1. Move sessions/audit/feedback to a shared database with conditional writes.
2. Move rate limits to Redis (or a gateway).
3. Make the index read-only and bake it into the image, persist the optional
   FAISS index (`VECTOR_STORE=faiss`, `INDEX_DIR`), or serve it from an external
   vector store.
4. Replace the in-memory tool server with a real service that keeps the same
   ownership checks and idempotency key.
5. Keep per-session ordering with a distributed lock, or accept eventual
   consistency and document it.

Do not add replicas while sessions and limits remain local; two replicas would
disagree about both.

## 11. Evaluation

`evals/run_evals.py` replays `evals/dataset.jsonl` from an empty memory per case
and checks the final outcome: outcomes, verified flag, required/forbidden
phrases, source bounds, LLM call caps, PII redaction count and tool results. It
writes `evals/report.json` and exits non-zero below `--min-pass`.

- Offline gate: 26 cases, 100% with `LLM_PROVIDER=mock` and
  `EMBEDDING_BACKEND=hash`.
- Cases with an empty `expect` object are refused, so an unlabelled feedback
  export cannot pass by accident.
- `evals/feedback_to_cases.py` turns thumbs-down traces into candidate cases; a
  human still writes the expectations.
- The dataset is small and synthetic. It is a regression gate, not a benchmark
  of real-world answer quality.

`evals/run_retrieval.py` is a separate retrieval gate. It builds the index with
whatever `EMBEDDING_BACKEND` / `VECTOR_STORE` / `INGEST_BACKEND` is configured,
ranks `evals/retrieval_cases.jsonl` (15 labelled queries) and reports
recall@1/@3/@5 and MRR. The same cases run against builtin and optional
backends, which is how backend parity is checked:

```bash
python -m evals.run_retrieval --min-recall 0.8
VECTOR_STORE=faiss INGEST_BACKEND=llamaindex python -m evals.run_retrieval
```

At the current seven-document corpus both paths score 100% on this set. That
mostly proves the gate itself works; it is not evidence that FAISS or
LlamaIndex improve answer quality at this scale.

`evals/benchmark_rag.py` isolates the vector store: synthetic normalised
vectors, build time and top-10 query latency. It is a measurement tool, not a
pass/fail gate.
