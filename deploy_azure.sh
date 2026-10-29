#!/usr/bin/env bash
# Deploy to Azure Container Apps. Prereqs: Azure CLI installed, `az login` done,
# and the .env file filled in. Run:  bash deploy_azure.sh
set -euo pipefail
set -a; source .env; set +a

RG="${RG:-support-agents-rg}"
APP="${APP:-agentic-support}"
LOCATION="${LOCATION:-centralindia}"

az group create --name "$RG" --location "$LOCATION"

# Builds the Dockerfile in Azure (no local Docker needed) and creates the app.
az containerapp up \
  --name "$APP" --resource-group "$RG" --location "$LOCATION" \
  --source . --ingress external --target-port 8000 \
  --env-vars \
    LLM_PROVIDER="$LLM_PROVIDER" \
    AZURE_OPENAI_API_KEY="$AZURE_OPENAI_API_KEY" \
    AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
    AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
    AZURE_OPENAI_API_VERSION="$AZURE_OPENAI_API_VERSION" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    EMBEDDING_BACKEND=hf

echo "Done. Your URL:"
az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv
echo "Test:  curl -X POST https://<url>/chat -H 'content-type: application/json' -d '{\"message\":\"my router keeps dropping\"}'"
