#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?}"
: "${API_IMAGE:?}"
: "${SOURCE_SHA:?}"
: "${DEPLOY_SHA:?}"
: "${CORPUS_VERSION_ID:?}"
: "${CORPUS_BUCKET_NAME:?}"
: "${CONFIGURATION_VERSION_ID:?}"
: "${RUNTIME_SECRET_VERSION:?}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$DEPLOY_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNTIME_SECRET_VERSION" =~ ^[1-9][0-9]*$ ]]

REGION=europe-west1
API_SERVICE=dmbok-compass-api
INGESTION_JOB=dmbok-compass-ingestion
EVALUATION_JOB=dmbok-compass-evaluation
PARSER_JOB=dmbok-compass-parser
COMMON_ENV="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},RAG_VECTOR_COLLECTION=document_chunks_v2,RAG_VECTOR_SCHEMA_VERSION=2,RAG_EMBEDDING_MODEL=gemini-embedding-001,RAG_EMBEDDING_DIMENSIONS=768,RAG_RETRIEVAL_STRATEGY=dense,RAG_CORPUS_VERSION_ID=${CORPUS_VERSION_ID},CONFIGURATION_VERSION_ID=${CONFIGURATION_VERSION_ID},ARTIFACT_ID=${ARTIFACT_ID:-},RAG_SCHEMA_VERSION=v2,RAG_VECTOR_INDEX=document_chunks_v2_vector,RUNTIME_SECRET_VERSION=${RUNTIME_SECRET_VERSION},SOURCE_COMMIT_SHA=${SOURCE_SHA},DEPLOYMENT_COMMIT_SHA=${DEPLOY_SHA}"
ANSWER_ENV="ANSWER_PROVIDER=openrouter,ANSWER_REQUEST_TIMEOUT_MS=14000,ANSWER_GENERATION_TIMEOUT_MS=8000,ANSWER_OUTPUT_TOKEN_CAP=1024,ANSWER_MAX_MODEL_ATTEMPTS=2,OPENROUTER_JSON_SCHEMA_SUPPORTED=true,OPENROUTER_PRIMARY_MODEL=meta-llama/llama-3.1-8b-instruct,OPENROUTER_FALLBACK_MODEL=google/gemma-4-26b-a4b-it:free"
SECRET="OPENROUTER_API_KEY=answer-provider-api-key:${RUNTIME_SECRET_VERSION}"
PARSER_ENV="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}"
if [[ -n "${PARSER_SECRET_VERSION:-}" ]]; then
  [[ "$PARSER_SECRET_VERSION" =~ ^[1-9][0-9]*$ ]]
  for setting in PARSER_ROUTE PARSER_MODE PARSER_ALLOWANCE_POLICY; do
    [[ "${!setting:-}" =~ ^[a-zA-Z0-9._/-]+$ ]]
  done
  [[ "${PARSER_ALLOWANCE_MAX_UNITS:-}" =~ ^[1-9][0-9]*$ ]]
  PARSER_ENV+=",PARSER_SECRET_VERSION=${PARSER_SECRET_VERSION},PARSER_ROUTE=${PARSER_ROUTE},PARSER_MODE=${PARSER_MODE},PARSER_ALLOWANCE_POLICY=${PARSER_ALLOWANCE_POLICY},PARSER_ALLOWANCE_MAX_UNITS=${PARSER_ALLOWANCE_MAX_UNITS}"
elif [[ -n "${PARSER_ROUTE:-}${PARSER_MODE:-}${PARSER_ALLOWANCE_POLICY:-}${PARSER_ALLOWANCE_MAX_UNITS:-}" ]]; then
  echo "Incomplete parser configuration" >&2
  exit 1
fi
EVALUATION_ENV="${COMMON_ENV},${ANSWER_ENV}"
for stage in GENERATOR JUDGE; do
  route="${stage}_ROUTE"
  model="${stage}_MODEL"
  version="${stage}_CONFIGURATION_VERSION"
  policy="${stage}_ALLOWANCE_POLICY"
  maximum="${stage}_ALLOWANCE_MAX_UNITS"
  if [[ -n "${!route:-}${!model:-}${!version:-}${!policy:-}${!maximum:-}" ]]; then
    for setting in "$route" "$model" "$version" "$policy"; do
      [[ "${!setting:-}" =~ ^[a-zA-Z0-9._/-]+$ ]]
      EVALUATION_ENV+=",${setting}=${!setting}"
    done
    [[ "${!maximum:-}" =~ ^[1-9][0-9]*$ ]]
    EVALUATION_ENV+=",${maximum}=${!maximum}"
  fi
done

PARSER_JOB_PRESENT=$(gcloud run jobs list --project="$PROJECT_ID" --region="$REGION" \
  --filter="metadata.name=${PARSER_JOB}" --format="value(metadata.name)" --limit=1 --quiet)
if [[ -n "${PARSER_SECRET_VERSION:-}" && "$PARSER_JOB_PRESENT" != "$PARSER_JOB" ]]; then
  echo "Parser job must be provisioned before enabling parsing" >&2
  exit 1
fi

gcloud run deploy "$API_SERVICE" --project="$PROJECT_ID" --region="$REGION" \
  --image="$API_IMAGE" --remove-secrets=LLAMACLOUD_API_KEY --update-secrets="$SECRET" \
  --update-env-vars="${COMMON_ENV},${ANSWER_ENV},CORPUS_BUCKET_NAME=${CORPUS_BUCKET_NAME},EVALUATION_JOB_NAME=${EVALUATION_JOB}" \
  --quiet

gcloud run jobs deploy "$INGESTION_JOB" --project="$PROJECT_ID" --region="$REGION" \
  --image="$API_IMAGE" --remove-secrets=LLAMACLOUD_API_KEY --update-secrets="$SECRET" \
  --update-env-vars="${COMMON_ENV},${ANSWER_ENV}" --quiet

gcloud run jobs deploy "$EVALUATION_JOB" --project="$PROJECT_ID" --region="$REGION" \
  --image="$API_IMAGE" --remove-secrets=LLAMACLOUD_API_KEY --update-secrets="$SECRET" \
  --remove-env-vars=GENERATOR_ROUTE,GENERATOR_MODEL,GENERATOR_CONFIGURATION_VERSION,GENERATOR_ALLOWANCE_POLICY,GENERATOR_ALLOWANCE_MAX_UNITS,JUDGE_ROUTE,JUDGE_MODEL,JUDGE_CONFIGURATION_VERSION,JUDGE_ALLOWANCE_POLICY,JUDGE_ALLOWANCE_MAX_UNITS \
  --update-env-vars="$EVALUATION_ENV" --quiet

if [[ "$PARSER_JOB_PRESENT" != "$PARSER_JOB" ]]; then
  PARSER_JOB_PRESENT=$(gcloud run jobs list --project="$PROJECT_ID" --region="$REGION" \
    --filter="metadata.name=${PARSER_JOB}" --format="value(metadata.name)" --limit=1 --quiet)
fi

if [[ -n "${PARSER_SECRET_VERSION:-}" ]]; then
  gcloud run jobs deploy "$PARSER_JOB" --project="$PROJECT_ID" --region="$REGION" \
    --image="$API_IMAGE" --update-secrets="LLAMACLOUD_API_KEY=llamacloud-parser-api-key:${PARSER_SECRET_VERSION}" \
    --update-env-vars="$PARSER_ENV" --quiet
elif [[ "$PARSER_JOB_PRESENT" == "$PARSER_JOB" ]]; then
  gcloud run jobs deploy "$PARSER_JOB" --project="$PROJECT_ID" --region="$REGION" \
    --image="$API_IMAGE" --remove-secrets=LLAMACLOUD_API_KEY \
    --remove-env-vars=PARSER_SECRET_VERSION,PARSER_ROUTE,PARSER_MODE,PARSER_ALLOWANCE_POLICY,PARSER_ALLOWANCE_MAX_UNITS \
    --update-env-vars="$PARSER_ENV" --quiet
fi
