"""FastAPI service: this is what gets deployed to Azure.

    uvicorn api:app --host 0.0.0.0 --port 8000
    POST /chat    {"session_id": "abc", "message": "my router keeps dropping"}
    GET  /health  liveness probe        GET /metrics  latency / token / cost summary
"""
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from agents_app.memory import ConversationMemory
from agents_app.workflow import SupportWorkflow

workflow = SupportWorkflow()
sessions: Dict[str, ConversationMemory] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await workflow.start()
    yield
    await workflow.stop()


app = FastAPI(title="Agentic Customer Support", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    memory = sessions.setdefault(req.session_id, ConversationMemory())
    return await workflow.handle(req.message, memory)


@app.get("/metrics")
async def metrics():
    return workflow.metrics.summary()
