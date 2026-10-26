#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?}"
firebase_dir="$(mktemp -d)"
trap 'rm -rf "$firebase_dir"' EXIT
cp -R web/dist "$firebase_dir/dist"
jq -n --arg service dmbok-compass-api --arg region europe-west1 \
  '{hosting: [{target: "app", public: "dist", rewrites: [{source: "/api/**", run: {serviceId: $service, region: $region}}]}]}' \
  > "$firebase_dir/firebase.json"
jq -n --arg project "$PROJECT_ID" \
  '{projects: {default: $project}, targets: {($project): {hosting: {app: [$project]}}}}' \
  > "$firebase_dir/.firebaserc"
(cd "$firebase_dir" && npx --yes firebase-tools@15.34.1 deploy --only hosting \
  --project="$PROJECT_ID" --non-interactive)
