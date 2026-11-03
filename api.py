"""Production HTTP API.

    uvicorn api:app --host 0.0.0.0 --port 8000

    POST   /v1/chat                 chat turn (X-API-Key required when API_KEYS is set)
    POST   /v1/chat/stream          same, as Server-Sent Events: live agent progress, then the answer
    POST   /v1/feedback             thumbs up/down on a trace_id
    DELETE /v1/sessions/{id}        erase a conversation + its audit trail (right to erasure)
    GET    /health/live             liveness probe
    GET    /health/ready            readiness probe (MCP, DB, index)
    GET    /metrics                 Prometheus metrics
    GET    /                        demo chat UI

Identity: the calling BACKEND authenticates the end customer, then sends X-Customer-Id. Tools are scoped to
that id server-side (an LLM can never choose it). Keep the API key on the server, never in a browser.
"""
import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from weakref import WeakValueDictionary

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST

from agents_app.config import settings
from agents_app.logging_setup import log_event, setup_logging
from agents_app.schemas import ChatRequest, ChatResponse, FeedbackRequest, SourceRef
from agents_app.security import SlidingWindowRateLimiter, key_fingerprint, redact_pii, verify_api_key
from agents_app.store import SessionStore
from agents_app.workflow import SupportWorkflow

log = logging.getLogger("support.api")
STATIC_DIR = Path(__file__).parent / "static"
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_CUSTOMER_RE = re.compile(r"[A-Za-z0-9_.@-]{1,64}")
_session_locks: "WeakValueDictionary[str, asyncio.Lock]" = WeakValueDictionary()


def _lock_for(key: str) -> asyncio.Lock:
    """One lock per session: two simultaneous messages in the same conversation are processed in order."""
    lock = _session_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[key] = lock
    return lock


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.validate_for_startup()                    # refuse to boot with unsafe production config
    store = SessionStore(settings.db_path)
    workflow = SupportWorkflow(store)
    await workflow.start()
    app.state.store, app.state.workflow = store, workflow
    app.state.limiter = SlidingWindowRateLimiter(settings.rate_limit_per_min)          # per customer
    app.state.key_limiter = SlidingWindowRateLimiter(settings.rate_limit_key_per_min)  # per API key (ceiling)
    app.state.bg_tasks = set()                          # strong refs so streaming turns can't be GC'd mid-flight

    async def retention_loop() -> None:
        while True:
            try:
                n = await store.purge_older_than(settings.retention_days)
                if n:
                    log_event(log, logging.INFO, "retention purge", rows=n)
            except Exception:                           # noqa: BLE001
                log.exception("retention purge failed")
            await asyncio.sleep(3600)

    task = asyncio.create_task(retention_loop())
    log_event(log, logging.INFO, "service started", environment=settings.environment,
              auth="on" if settings.api_keys else "OFF (dev only)")
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await workflow.stop()
        store.close()


MAX_BODY_BYTES = 64 * 1024


class BodyLimitMiddleware:
    """Reject oversized request bodies BEFORE they are read into memory and parsed. Covers both a declared
    Content-Length and chunked uploads that declare nothing."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app, self.max_bytes = app, max_bytes

    @staticmethod
    async def _send_413(send) -> None:
        body = b'{"error":"payload_too_large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length and length.isdigit() and int(length) > self.max_bytes:
            return await self._send_413(send)

        seen, exceeded, answered = 0, False, False

        async def limited_receive():
            nonlocal seen, exceeded
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    exceeded = True
                    raise _TooLarge()          # stop reading: memory use stays bounded
            return message

        async def guarded_send(message):
            # The framework turns our exception into its own generic 400; replace that with the accurate 413.
            nonlocal answered
            if exceeded:
                if message["type"] == "http.response.start" and not answered:
                    answered = True
                    await self._send_413(send)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _TooLarge:
            if not answered:
                await self._send_413(send)


class _TooLarge(Exception):
    pass


app = FastAPI(title="Agentic Customer Support", version="2.2.0", lifespan=lifespan, redoc_url=None,
              docs_url=None if settings.is_production else "/docs",
              openapi_url=None if settings.is_production else "/openapi.json")

app.add_middleware(BodyLimitMiddleware)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["x-api-key", "content-type"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    rid = (request.headers.get("x-request-id") or uuid.uuid4().hex[:16])[:64]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:                                   # noqa: BLE001  never leak internals to clients
        log.exception("unhandled error")
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if not request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    if request.url.path == "/":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
    if request.url.path not in ("/health/live", "/health/ready", "/metrics"):
        log_event(log, logging.INFO, "http", method=request.method, path=request.url.path,
                  status=response.status_code, ms=round((time.perf_counter() - start) * 1000, 1), request_id=rid)
    return response


async def authenticate(request: Request, x_api_key: Optional[str] = Header(default=None)) -> str:
    """API-key auth (constant-time compare) + per-identity rate limiting. Returns the caller identity."""
    if settings.api_keys:
        key = verify_api_key(x_api_key, settings.api_keys)
        if key is None:
            raise HTTPException(401, "invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"})
        identity = key_fingerprint(key)
    else:
        identity = "dev"
    # Two buckets. In the intended architecture ONE backend key serves ALL customers, so a per-key limit alone
    # would throttle the whole product. Limit per customer (fairness) and per key (a ceiling on total load).
    customer = request.headers.get("x-customer-id") or settings.default_customer_id or "anonymous"
    if not _CUSTOMER_RE.fullmatch(customer):
        customer = "invalid"
    for limiter, bucket in ((request.app.state.key_limiter, identity),
                            (request.app.state.limiter, f"{identity}:{customer}")):
        allowed, retry_after = limiter.allow(bucket)
        if not allowed:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry_after)})
    return identity


# --------------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health/live", include_in_schema=False)
async def live(request: Request):
    """Liveness includes the tool-server subprocess. If it dies the process cannot heal itself, and with a
    single replica a readiness-only failure would mean a permanent outage. Failing liveness makes the platform
    restart the container, which is the self-heal."""
    if not await request.app.state.workflow.mcp.ping():
        return JSONResponse({"status": "tool_server_unreachable"}, status_code=503)
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
async def ready(request: Request):
    report = await request.app.state.workflow.readiness()
    return JSONResponse(report, status_code=200 if report["ready"] else 503)


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request, _: str = Depends(authenticate)):
    return Response(request.app.state.workflow.metrics.prometheus_text(), media_type=CONTENT_TYPE_LATEST)


async def customer_dep(x_customer_id: Optional[str] = Header(default=None)) -> str:
    """Who the end customer is, as asserted by the authenticated calling backend."""
    if x_customer_id is None:
        return settings.default_customer_id            # demo default; set EMPTY in production
    if not _CUSTOMER_RE.fullmatch(x_customer_id):
        raise HTTPException(422, "invalid X-Customer-Id")
    return x_customer_id


def _session_key(identity: str, customer_id: str, session_id: str) -> str:
    """Sessions are namespaced by API key AND customer: one customer can never reach another's history."""
    return f"{identity}:{customer_id or 'anonymous'}:{session_id}"


async def _run_turn(request: Request, req: ChatRequest, identity: str, customer_id: str, on_event=None) -> dict:
    wf: SupportWorkflow = request.app.state.workflow
    store: SessionStore = request.app.state.store
    key = _session_key(identity, customer_id, req.session_id)
    async with _lock_for(key):                         # messages in one conversation are handled in order
        memory = await store.load(key)
        result = await wf.handle(req.message, memory, session_key=key, customer_id=customer_id, on_event=on_event)
        await store.save(key, memory)
    return result


def _include_debug() -> bool:
    return settings.debug_responses and not settings.is_production


def _to_response(result: dict) -> ChatResponse:
    return ChatResponse(
        trace_id=result["trace_id"], answer=result["answer"], outcome=result["outcome"],
        verified=result["verified"], sources=[SourceRef(**s) for s in result["sources"]],
        needs_confirmation=result["needs_confirmation"],
        debug={**result["debug"], "tool_calls": result["tool_calls"]} if _include_debug() else None)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, identity: str = Depends(authenticate),
               customer_id: str = Depends(customer_dep)):
    return _to_response(await _run_turn(request, req, identity, customer_id))


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, identity: str = Depends(authenticate),
                      customer_id: str = Depends(customer_dep)):
    """Server-Sent Events. `step` events show the agents working (perceived latency drops a lot), then ONE
    `answer` event carries the verified reply. We deliberately do not stream raw model tokens: the draft is
    fact-checked first, and streaming unverified text would defeat the verifier."""
    queue: asyncio.Queue = asyncio.Queue()
    detailed = _include_debug()

    def on_event(kind: str, data: dict) -> None:
        if kind == "step" and not detailed:            # production: coarse stage names only, no internals
            data = {"stage": data["text"].split(" ")[0].split("->")[0].strip()}
        queue.put_nowait((kind, data))

    async def runner() -> None:
        try:
            result = await _run_turn(request, req, identity, customer_id, on_event)
            queue.put_nowait(("answer", json.loads(_to_response(result).model_dump_json())))
        except Exception:                               # noqa: BLE001
            log.exception("stream turn failed")
            queue.put_nowait(("error", {"error": "internal_error"}))
        finally:
            queue.put_nowait(None)

    request.app.state.bg_tasks.add(task := asyncio.create_task(runner()))   # survives client disconnects
    task.add_done_callback(request.app.state.bg_tasks.discard)

    async def events():
        while (item := await queue.get()) is not None:
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest, request: Request, _: str = Depends(authenticate)):
    comment, _ = redact_pii(req.comment)                    # free text: users type emails and phone numbers
    ok = await request.app.state.store.add_feedback(req.trace_id, req.rating, comment)
    if not ok:
        raise HTTPException(404, "unknown trace_id")
    request.app.state.workflow.metrics.count_feedback(req.rating)
    return {"status": "recorded"}


@app.delete("/v1/sessions/{session_id}")
async def erase_session(session_id: str, request: Request, identity: str = Depends(authenticate),
                        customer_id: str = Depends(customer_dep)):
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise HTTPException(422, "invalid session id")
    n = await request.app.state.store.delete_session(_session_key(identity, customer_id, session_id))
    return {"deleted_rows": n}
