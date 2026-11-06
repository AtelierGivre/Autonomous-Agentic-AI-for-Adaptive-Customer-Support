
# Autonomous Agentic AI for Adaptive Customer Support

An agentic AI customer support system that uses multiple specialized agents to understand customer queries, retrieve relevant information, interact with tools, troubleshoot issues, and verify responses.

The system is designed to provide context-aware support while incorporating safeguards for hallucinations, customer data isolation, ambiguous consent, prompt injection, and sensitive information.

## Architecture

The system follows a multi-agent workflow:

**Planner → Retriever + Tool Agent → Troubleshooter → Verifier**

- **Planner:** Understands the customer's request and determines the steps required to handle it.
- **Retriever:** Retrieves relevant information from the knowledge base.
- **Tool Agent:** Interacts with available tools through MCP.
- **Troubleshooter:** Uses the retrieved information and tool results to develop a solution.
- **Verifier:** Checks the generated response before it is returned to the customer.

The workflow can adapt its actions based on the conversation context. Its agent plan is autonomous, but it does not autonomously modify its own production code or behavior.

## Technology Stack

- **Language:** Python
- **LLM orchestration:** LangChain
- **RAG and indexing:** LlamaIndex, FAISS
- **Models:** Hugging Face Transformers
- **Tool integration:** Model Context Protocol (MCP)
- **Backend:** FastAPI
- **Cloud:** Microsoft Azure

The application supports cloud-based LLMs, local Hugging Face models, and local OpenAI-compatible model servers.

The default local model is `Qwen/Qwen2.5-3B-Instruct`.

Local OpenAI-compatible servers such as Ollama, vLLM, and llama.cpp can also be configured.

## Key Features

### 1. Multi-Agent Customer Support

Specialized agents handle planning, information retrieval, tool execution, troubleshooting, and response verification.

### 2. Retrieval-Augmented Generation

FAISS is used for vector search over the knowledge base. Retrieved information provides context for generating customer support responses.

### 3. MCP Tool Integration

The tool agent uses the Model Context Protocol to interact with external tools.

### 4. Adaptive Conversations

The system uses conversation context to adapt its actions and responses during a support session.

### 5. Response Verification

The verifier checks generated responses against available information. If a response needs correction, one revision is allowed.

### 6. Multiple LLM Providers

The system supports:
- Cloud-based LLM providers.
- Local Hugging Face models.
- Local OpenAI-compatible model servers.
- An offline mock LLM for testing.

## Security and Reliability

### Hallucination Control

- Uses a retrieval confidence threshold.
- Responses with insufficient retrieval confidence can be rejected.
- The verifier checks the generated response against available information.
- One revision is allowed when verification identifies an issue.

### Customer Data Isolation

- Customer identity is injected into tools by the server.
- The model cannot select or modify the customer identity.
- Requests for another customer's order are handled as not found.

### Consent for Side-Effecting Actions

- Actions that can cause side effects are proposed rather than automatically executed.
- Confirmation must be an unambiguous yes or no.
- Confirmation expires to prevent stale approvals from being reused.

### Circuit Breaker

- Tracks actual service failures, including timeouts, HTTP 429 responses, and HTTP 5xx responses.
- Content-filter rejections do not count as service outages.

### PII Protection

Sensitive information is redacted before reaching the model, memory, or audit records.

Detection includes:
- Email addresses.
- Phone numbers.
- Payment card numbers validated using the Luhn algorithm.
- Aadhaar/PAN-like identifiers.
- Secrets.

The system also includes retention purge and erasure functionality.

### Prompt Injection Protection

- Screens incoming user input.
- Treats retrieved and externally supplied text as untrusted.
- Fences or sanitizes untrusted content.
- Excludes suspicious documents.
- Restricts tool arguments using schemas and allow-lists.

## Project Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Configure the required provider settings in `.env` for your chosen LLM provider.

### 5. Run the Application

Run the application:

```bash
python main.py
```

Start the API:

```bash
uvicorn api:app --port 8000
```

## Running Offline

The project includes an offline mock LLM and hash-based embeddings.

Run:

```bash
LLM_PROVIDER=mock EMBEDDING_BACKEND=hash MIN_RETRIEVAL_SCORE=0.1 python main.py
```

Alternatively:

```bash
make offline
```

This mode allows the application and its deterministic controls to be tested without external model API credentials.

## Running with a Local Model

Set the provider to local:

```bash
LLM_PROVIDER=local
```

Run:

```bash
make local
```

The default local model is:

```text
Qwen/Qwen2.5-3B-Instruct
```

For a local OpenAI-compatible server, configure:

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b-instruct
```

Run local evaluations:

```bash
make eval-local
```

## Demo Scenarios

The application includes sample scenarios for testing customer support behavior.

| Scenario | Example |
|---|---|
| Connectivity issue | Router disconnecting |
| Order lookup | ORD-1003 |
| Escalation | Customer requests escalation |
| Prompt injection | Malicious instructions in user input |
| Out-of-scope query | Weather-related question |
| Low-confidence retrieval | Quantum flux capacitor |
| PII redaction | Email address in a message |
| Customer isolation | Unauthorized lookup of ORD-9001 |

### Sample Order Data

| Order ID | Status |
|---|---|
| ORD-1001 | Delivered |
| ORD-1002 | In transit |
| ORD-1003 | Delayed |

The interface includes a **What the agents did** section to show the actions taken during the conversation.

Metrics are available through:

```text
/metrics
```

## Testing and Evaluation

Run linting:

```bash
ruff check .
```

Run tests:

```bash
pytest -q
```

Run the evaluation suite:

```bash
python -m evals.run_evals --min-pass 0.9
```

Run the combined checks:

```bash
make check
```

Feedback can be exported into evaluation cases using:

```bash
python -m evals.feedback_to_cases
```

Evaluation thresholds include:
- `answerable_min`
- `unanswerable_max`

The project includes an offline mock LLM to test deterministic controls without requiring external API credentials.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/chat` | Send a customer support message |
| POST | `/v1/chat/stream` | Stream a response using SSE |
| POST | `/v1/feedback` | Submit response feedback |
| DELETE | `/v1/sessions/{id}` | Delete a session |
| GET | `/health/live` | Liveness check |
| GET | `/health/ready` | Readiness check |
| GET | `/metrics` | Application metrics |

### Authentication Headers

The API supports:

- `X-API-Key`
- `X-Customer-Id`

The backend must authenticate the customer and establish the customer identity before forwarding requests.


The demo UI uses the `demo` key.

### Production Configuration

Production configuration uses:

```env
ENVIRONMENT=production
```

Production mode requires `API_KEYS`, rejects the mock provider, and hides debug information and API documentation.

## Azure Deployment

The project includes an Azure deployment script.

Authenticate with Azure:

```bash
az login
```

Run:

```bash
bash deploy_azure.sh
```

The deployment script is intended to deploy the application using Docker and Azure Container Apps.

## Project Structure

```text
.
├── agents_app/
│   ├── agents.py
│   ├── workflow.py
│   ├── rag.py
│   ├── mcp_server.py
│   ├── mcp_client.py
│   ├── llm.py
│   ├── resilience.py
│   ├── security.py
│   ├── guardrails.py
│   ├── store.py
│   ├── memory.py
│   ├── monitoring.py
│   ├── logging_setup.py
│   └── api.py
├── static/
├── evals/
├── tests/
├── docs/
├── knowledge_base/
│   └── *.md
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```
