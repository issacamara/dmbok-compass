#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?}"
: "${API_IMAGE:?}"
: "${SOURCE_SHA:?}"
: "${DEPLOY_SHA:?}"
: "${CORPUS_VERSION_ID:?}"
: "${RUNTIME_SECRET_VERSION:?}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$DEPLOY_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNTIME_SECRET_VERSION" =~ ^[1-9][0-9]*$ ]]

REGION=europe-west1
API_SERVICE=dmbok-compass-api
INGESTION_JOB=dmbok-compass-ingestion
EVALUATION_JOB=dmbok-compass-evaluation
COMMON_ENV="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},RAG_VECTOR_COLLECTION=document_chunks_v2,RAG_VECTOR_SCHEMA_VERSION=2,RAG_EMBEDDING_MODEL=gemini-embedding-001,RAG_EMBEDDING_DIMENSIONS=768,RAG_RETRIEVAL_STRATEGY=dense,RAG_CORPUS_VERSION_ID=${CORPUS_VERSION_ID},RAG_SCHEMA_VERSION=v2,RAG_VECTOR_INDEX=document_chunks_v2_vector,RUNTIME_SECRET_VERSION=${RUNTIME_SECRET_VERSION},SOURCE_COMMIT_SHA=${SOURCE_SHA},DEPLOYMENT_COMMIT_SHA=${DEPLOY_SHA}"
ANSWER_ENV="ANSWER_PROVIDER=openrouter,ANSWER_REQUEST_TIMEOUT_MS=14000,ANSWER_GENERATION_TIMEOUT_MS=8000,ANSWER_OUTPUT_TOKEN_CAP=1024,ANSWER_MAX_MODEL_ATTEMPTS=2,OPENROUTER_JSON_SCHEMA_SUPPORTED=true,OPENROUTER_PRIMARY_MODEL=meta-llama/llama-3.1-8b-instruct,OPENROUTER_FALLBACK_MODEL=google/gemma-4-26b-a4b-it:free"
SECRET="OPENROUTER_API_KEY=answer-provider-api-key:${RUNTIME_SECRET_VERSION}"

gcloud run deploy "$API_SERVICE" --project="$PROJECT_ID" --region="$REGION" \
  --image="$API_IMAGE" --update-secrets="$SECRET" \
  --update-env-vars="${COMMON_ENV},${ANSWER_ENV},EVALUATION_JOB_NAME=${EVALUATION_JOB}" \
  --quiet

gcloud run jobs deploy "$INGESTION_JOB" --project="$PROJECT_ID" --region="$REGION" \
  --image="$API_IMAGE" --update-secrets="$SECRET" \
  --update-env-vars="${COMMON_ENV},${ANSWER_ENV}" --quiet

gcloud run jobs deploy "$EVALUATION_JOB" --project="$PROJECT_ID" --region="$REGION" \
  --image="$API_IMAGE" --update-secrets="$SECRET" \
  --update-env-vars="${COMMON_ENV},${ANSWER_ENV}" --quiet
