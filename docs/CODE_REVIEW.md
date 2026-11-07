# Code review log

This project treats generated code as untrusted. The review below is the record
of what was checked, what was wrong, and which test prevents a regression.

## Context

The published repository documented a full architecture in `README.md`,
`.env.example`, `Dockerfile`, `Makefile` and `api.py`, but the tree did not
contain the `agents_app` package, the knowledge base, the demo UI, the tests or
the eval harness. Importing `api:app` failed immediately.

The correction was therefore a reconstruction against the documented contracts,
not a small patch:

- modules were written to match the imports and interfaces in `api.py`,
  `main.py` and the README;
- every claim in the docs was re-checked against behaviour;
- dependencies that were listed but never used were removed;
- the end-to-end pipeline was run and measured, not assumed.

That context matters for the findings: “the code was wrong” here often means
“the code did not exist, and the first draft had a real defect”.

## Findings

| # | Severity | Finding | Fix | Regression test |
|---|---|---|---|---|
| 1 | **Critical** | Published tree could not start: `agents_app`, `knowledge_base`, `static` and tests were missing. | Reconstructed the packages and wired them to `api.py`/`main.py`. | `pytest -q` imports and exercises every module; `test_api.py` boots the app. |
| 2 | **High** | `requirements.txt` listed LangChain, LlamaIndex, FAISS and `mcp`, none of which the implementation used. A fresh install pulled unused heavy packages and could fail on Python 3.14. | Trimmed runtime deps; moved Transformers and sentence-transformers to `requirements-local.txt`; updated Dockerfile and Makefile. | Clean install + `pytest -q` with no torch and no network. |
| 3 | **High** | Duplicate ticket IDs were possible: the sequence increment and the duplicate check were not one atomic step, so two concurrent confirmations could interleave. | One lock covers the duplicate lookup, sequence increment and append. Tickets are idempotent per `(customer, subject)`. | `test_mcp.py::test_ticket_creation_is_idempotent_per_customer` |
| 4 | **High** | Output filtering was incomplete: it did not reliably remove both `<untrusted>` markers and did not bound the reply size, so internal prompt markers could reach a customer. | `output_guard()` strips both markers and caps at 4000 characters; the verifier sees the same text the customer will see. | `test_guardrails.py::test_output_guard_strips_markers_and_caps_length` |
| 5 | **High** | Provider failures (outage, exhausted retries, open breaker, content-filter rejection) surfaced as a raw HTTP 500. | `LLMError` around planning and drafting now returns outcome `degraded` with a safe message and no side effects. Only real outages still trip the breaker. | `test_workflow.py::test_provider_failure_degrades_to_a_safe_reply` |
| 6 | **Medium** | The eval harness accepted a feedback candidate with `expect: {}`; because no checks ran, it passed automatically. | `load_cases()` refuses any case without expectations. | `test_evals.py::test_golden_dataset_loads_and_unlabelled_cases_are_refused` |
| 7 | **Medium** | The documented local-model knobs were only partly wired: `LOCAL_MODEL_FAST`, `LOCAL_MODEL_REVISION`, `LOCAL_DTYPE` and the non-auto `LOCAL_DEVICE` values were read but ignored. | Passed through to `transformers.pipeline`; pipelines cached per model; `accelerate` added for `device_map="auto"`. | Configuration review; not exercised in CI because it downloads multi-GB weights. |
| 8 | **Medium** | Retrieval answered “charged twice” with the generic card-statement sentence and missed the 12-month warranty term: the score blended rank position but not whether the chunk actually contained the asked-about terms. | Query synonyms, intent priors, coverage-aware scoring, and sentence selection ordered by shared terms first. | `evals/dataset.jsonl`: `duplicate-charge`, `hardware-warranty` |
| 9 | **Low** | Docs claimed 153 tests, 32 eval cases, a FAISS index and a subprocess MCP server. None were true. | README, Makefile, Dockerfile and comments now describe the measured stack and counts. | Manual doc sweep for `FAISS`, `LangChain`, `LlamaIndex`, `153`, `32 cases`. |
| 10 | **Low** | Feedback export could copy PII into the eval corpus. | The last customer turn and the free-text comment are redacted again on export. | Manual end-to-end export from a temporary SQLite database. |

## Representative fixes in detail

### 3. Duplicate tickets

The unsafe shape is:

```python
# do not do this
ticket_id = f"TCK-{self._ticket_seq + 1}"
if already_exists(...):
    return existing
self._ticket_seq += 1
self._tickets.append(...)
```

Two coroutines can both read the same sequence, both pass the duplicate check,
and both append a ticket with the same ID. The fixed version puts the lookup,
increment and append inside one `threading.Lock`, because the tool server is
called from the event loop and the operation is short and synchronous:

```python
with self._lock:
    for ticket in self._tickets:
        if ticket["customer_id"] == customer_id and ticket["subject"] == subject:
            return {**ticket, "duplicate": True}
    self._ticket_seq += 1
    ...
```

The idempotency key is `(customer, subject)`, so a retried confirmation or a
double-click reuses the existing ticket instead of opening a second one. The
response carries `duplicate: true`, and the workflow tells the customer the
ticket already existed.

### 4. Output filter

A model or a mock can echo the fences used to mark untrusted input. Returning
that text leaks internal prompt structure and can confuse a downstream UI. The
guard is deliberately small and testable:

```python
def output_guard(text: str, max_chars: int = 4000) -> str:
    text = (text or "").replace("<untrusted>", "").replace("</untrusted>", "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text
```

The test asserts both properties: no marker survives, and the result is never
longer than the cap. The cap is not a security boundary on its own; the request
body limit and input screen are.

### 5. Degradation instead of a 500

`LLM._call_provider()` retries once, then raises `LLMError`. Before the fix,
that exception escaped `SupportWorkflow._handle()` and the API returned a
generic 500. Two failure classes needed different treatment:

- **Outages** (timeouts, 429, 5xx) count toward the circuit breaker.
- **Content-filter rejections** are not outages; they must not open the
  breaker, but they still must not become a 500.

Both now produce the same safe, non-committal reply with outcome `degraded`.
The planner and the draft/verify stage are wrapped independently, so a failure
before any tool call cannot create a ticket, and a failure after retrieval
cannot ship an unverified draft.

### 8. Retrieval quality

The first scoring formula ranked a chunk highly for sharing a common word even
when it lacked the specific term the customer used. Two probes exposed it:

- “I was charged twice” returned the generic statement sentence, not the
  duplicate-refund procedure.
- “What does the hardware warranty cover?” omitted `12-month`.

The fix was not a larger model. It was:

- stem the query and document tokens consistently;
- expand a small, explicit thesaurus (`twice`/`double` → `duplicate`);
- add a small intent prior so billing questions see `billing.md` first;
- include query-token coverage in the score;
- choose answer sentences by shared content terms first, then document order.

Both probes are now eval cases, so the regression is measured on every run.

## Review checklist used

For each module:

- **Trust boundaries:** where does untrusted text enter, and is it fenced or
  redacted before the model sees it?
- **Authorization:** is the customer ID supplied by the caller and checked
  inside the tool, never by the model?
- **Side effects:** can any path execute a write without an unambiguous
  confirmation?
- **Failure modes:** what happens on timeout, provider error, bad JSON, empty
  retrieval, tool error, and budget exhaustion?
- **Determinism:** can the behaviour be tested offline with a forced mock?
- **Data lifecycle:** is anything stored that should have been redacted, and can
  it be erased?
- **Docs:** does every number and package name match the code that exists?

## Verification commands

```bash
ruff check .
pytest -q
python -m evals.run_evals --min-pass 0.9
```

The offline gate uses `LLM_PROVIDER=mock`, `EMBEDDING_BACKEND=hash`,
`MIN_RETRIEVAL_SCORE=0.1` and `MIN_LEXICAL_COVERAGE=0.5`. CI runs the same
commands with no secrets. A live smoke test is also expected before a release:

```bash
uvicorn api:app --port 8000
# GET /health/live, /health/ready, /metrics
# POST /v1/chat, /v1/chat/stream, /v1/feedback
```

## Open review items

These are known and not fixed here:

- The input screen is a small regex list. Paraphrased or encoded injection can
  pass; the fenced prompt and tool allow-list are the next layers, not a proof.
- PII detection misses names, postal addresses and many national ID formats.
- Rate limiting is in-process; a multi-replica deployment needs a shared store.
- The demo tool server keeps tickets in memory and resets on restart.
- The eval set is 26 synthetic cases. It catches regressions, not real-world
  coverage.
- The local-model path has no automated test because it downloads several GB of
  weights; it is implemented against the documented `transformers` contract but
  was not run in this workspace. Treat it as unvalidated until you measure it.
