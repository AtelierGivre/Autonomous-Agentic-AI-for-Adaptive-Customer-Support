#!/usr/bin/env bash
# Deploy to Azure Container Apps with secrets kept as Container App secrets (never plain env vars).
# Prereqs: Azure CLI, `az login`, .env filled in (Azure OpenAI section). Run:  bash deploy_azure.sh
# NOTE: not yet tested against a real Azure subscription; review before first use.
set -euo pipefail
[ -f .env ] || { echo "Missing .env. Run: cp .env.example .env  and fill in the Azure OpenAI section."; exit 1; }
set -a; source .env; set +a

RG="${RG:-support-agents-rg}"
APP="${APP:-agentic-support}"
LOCATION="${LOCATION:-centralindia}"

: "${AZURE_OPENAI_API_KEY:?fill AZURE_OPENAI_API_KEY in .env}"
: "${AZURE_OPENAI_ENDPOINT:?fill AZURE_OPENAI_ENDPOINT in .env}"
: "${AZURE_OPENAI_DEPLOYMENT:?fill AZURE_OPENAI_DEPLOYMENT in .env}"

# Generate an API key for callers if none is configured.
if [ -z "${API_KEYS:-}" ]; then
  API_KEYS="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "Generated API key for clients (save it!): $API_KEYS"
fi

az group create --name "$RG" --location "$LOCATION" >/dev/null

# 1) Build the Dockerfile in Azure and create the app (no local Docker needed).
az containerapp up --name "$APP" --resource-group "$RG" --location "$LOCATION" \
  --source . --ingress external --target-port 8000

# 2) Store secrets as Container App secrets.
az containerapp secret set -n "$APP" -g "$RG" --secrets \
  api-keys="$API_KEYS" azure-openai-key="$AZURE_OPENAI_API_KEY"

# 3) Production configuration. ONE replica: sessions are in SQLite and rate limits in memory.
#    To scale out, move both to Redis/Postgres first (docs/ARCHITECTURE.md, "Scaling").
az containerapp update -n "$APP" -g "$RG" --min-replicas 1 --max-replicas 1 \
  --set-env-vars \
    ENVIRONMENT=production LOG_JSON=true DEBUG_RESPONSES=false \
    LLM_PROVIDER=azure EMBEDDING_BACKEND=hf \
    API_KEYS=secretref:api-keys \
    AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
    AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
    AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
    AZURE_OPENAI_DEPLOYMENT_FAST="${AZURE_OPENAI_DEPLOYMENT_FAST:-}" \
    AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-10-21}"

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo
echo "Deployed: https://$FQDN"
echo "Probes to configure in the portal:  liveness /health/live   readiness /health/ready  (port 8000)"
echo "Persist chat history across restarts: mount an Azure Files volume at /app/data"
echo "Smoke test:"
echo "  curl -s -X POST https://$FQDN/v1/chat -H 'content-type: application/json' -H 'X-API-Key: <key>' \\"
echo "       -d '{\"session_id\":\"t1\",\"message\":\"my router keeps dropping\"}'"
