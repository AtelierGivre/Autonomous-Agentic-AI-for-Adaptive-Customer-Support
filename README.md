# Autonomous Agentic AI for Adaptive Customer Support

A working multi-agent customer-support system.
**Stack:** Python · LangChain · LlamaIndex · Hugging Face · FAISS · MCP · Azure

## 1. Run it (3 steps)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# open .env and fill in your key (see below), then:
python main.py
```

**The only thing you must do:** open `.env` and fill in ONE of:
- Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`
- or OpenAI: `OPENAI_API_KEY`

First run downloads a small embedding model (~130 MB) from Hugging Face once. No key needed for that.

**No key yet?** Test the plumbing offline: `LLM_PROVIDER=mock EMBEDDING_BACKEND=hash python main.py`
(a fake LLM, so answers are canned, but you can see planning, MCP tools, retrieval, memory, metrics).
Automated check: `python tests/smoke_test.py`

## 2. Demo script for the class

| You type | What students see |
|---|---|
| `My router keeps disconnecting` | Planner -> Retriever -> Troubleshooter -> Verifier. Answer cites `wifi_connectivity.md` |
| `Where is my order ORD-1003?` | Planner adds a **tool** step; the Tool agent calls `lookup_order` over **MCP** |
| `Still not working, escalate please` | **Memory**: remembers router + order; Tool agent creates a ticket |
| `/metrics` | Latency, tokens and **cost per agent** |

Valid fake orders: `ORD-1001` (delivered), `ORD-1002` (in transit), `ORD-1003` (delayed).

## 3. How it works

```
Customer message
      |
  PLANNER agent  ---- reads memory, writes a step list as JSON (dynamic task planning)
      |
      +--> RETRIEVER agent --> LlamaIndex + FAISS + HF embeddings --> help articles
      +--> TOOL agent      --> MCP client --> MCP server --> order / status / ticket tools
      |
  TROUBLESHOOTER agent --- writes the step-by-step reply from retrieved evidence only
      |
  VERIFIER agent --- checks the reply is grounded; if not, one revision (anti-hallucination)
      |
  MEMORY (last 6 turns + remembered facts) + MONITORING (latency / tokens / cost)
```

| File | Teaches |
|---|---|
| `agents_app/agents.py` | One class per agent, each with a single job and its own prompt |
| `agents_app/workflow.py` | Orchestration, dynamic re-planning, verify-and-revise loop |
| `agents_app/rag.py` | RAG: chunking, embeddings, FAISS index, persistence |
| `agents_app/mcp_server.py` / `mcp_client.py` | MCP: tools exposed over a standard protocol and discovered at runtime |
| `agents_app/memory.py` | Short-term window memory + fact memory |
| `agents_app/monitoring.py` | Latency tracking, token counting, cost estimation (also logged to `logs/metrics.jsonl`) |
| `agents_app/llm.py` | LangChain Azure/OpenAI wrapper (one place for all metrics) |
| `knowledge_base/*.md` | The "company docs". Edit or add files, delete `storage/`, restart to re-index |
| `api.py` | FastAPI service (`/chat`, `/health`, `/metrics`) for deployment |

## 4. Run as an API

```bash
uvicorn api:app --port 8000
curl -X POST localhost:8000/chat -H "content-type: application/json" \
     -d '{"session_id":"s1","message":"my router keeps dropping"}'
```
Each `session_id` gets its own memory.

## 5. Deploy to Azure

```bash
az login
bash deploy_azure.sh        # builds the Dockerfile in Azure Container Apps
```
Edit `RG`, `APP`, `LOCATION` at the top of the script if you want different names.
For production, store keys in Azure Key Vault / Container App secrets instead of plain env vars.

## 6. Cost and latency ideas (good discussion topics)

- Use a cheaper model for planner/verifier and a stronger one for the troubleshooter.
- Skip the verifier for simple FAQs; cache repeated questions.
- Lower `top_k` in `config.py` to send fewer tokens.
- Update `PRICE_INPUT_PER_1K` / `PRICE_OUTPUT_PER_1K` in `.env` to your model's real price.

## 7. Troubleshooting

- "No API key found": fill `.env` (or use `LLM_PROVIDER=mock`).
- Changed the knowledge base but answers are old: delete the `storage/` folder.
- Windows: if the MCP subprocess fails, run from the project root so `agents_app` is importable.
