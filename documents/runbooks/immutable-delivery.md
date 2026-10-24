# Immutable preview and production delivery

The GitHub Actions workflow in `.github/workflows/delivery.yml` builds the API
once, pushes it under the commit SHA, resolves the registry digest, and records
that digest in `delivery-manifest.json`. The same digest is used for the
zero-traffic Cloud Run revision. The frontend is tarred and hashed once; the
preview and production hosting steps consume that same uploaded artifact.
`delivery.tfplan.txt` is the human-review evidence for the Terraform plan used
by the workflow.

## Preview

Pushes to `main` target the `development` environment: they create a Cloud Run
revision with zero traffic and update the development Firebase Hosting site.
They do not route API traffic to the new revision or change production. Manual
runs may target either environment. Save the generated workflow artifact
containing `delivery-manifest.json`, the Terraform plan, preview URL, and
evaluation result with the release decision.

The manifest is valid only when `api_image` ends in `@api_digest`; a mutable
tag is not a promotable artifact. Do not rebuild between preview and promotion.

## Promotion

After the sponsor records the release decision and all evaluation gates pass,
store `release-evidence.json`, `release-manifest.json`, and
`release-decision.json` in the private release-evidence bucket at
`release-bundles/<release-decision-id>/`. The delivery identity has read-only
access; an authorized operator uploads the bundle. The promotion workflow
downloads those records and validates them against the exact preview artifact
before changing Firebase Hosting or Cloud Run traffic. Production additionally
requires approval from the protected `production` GitHub environment.

Run `Promote reviewed release` manually with the target environment, successful
preview run ID, and release decision ID. There is no automatic promotion. The
current `documents/release/langchain/release-evidence.json` is blocked, so it
cannot be used to promote a release until valid evidence and the sponsor's
recorded approval replace that blocked state.

## Rollback drill

1. Record the active Cloud Run revision and Firebase release before a drill.
2. Route Cloud Run traffic to the immediately prior known-good revision:
   `gcloud run services update-traffic SERVICE --region=europe-west1 --to-revisions=REVISION=100`.
3. Restore the prior Firebase Hosting release using the Firebase console or
   the documented release identifier; do not rebuild the frontend.
4. If the corpus is implicated, use the application’s immutable corpus-version
   rollback operation and verify the active-version pointer before serving
   traffic.
5. Run the smoke and citation checks, record timestamps and revision IDs, and
   return traffic only after the release approver accepts the evidence.

The drill passes only when the prior API revision, frontend release, and (when
needed) corpus version are restored without rebuilding or mutating an artifact.
