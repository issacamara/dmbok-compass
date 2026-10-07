# Immutable preview and production delivery

The Cloud Build manifest in `cloudbuild.yaml` builds the API once, pushes it
under the commit SHA, resolves the registry digest, and records that digest in
`delivery-manifest.json`. The same digest is used for the zero-traffic Cloud Run
revision. The frontend is tarred and hashed once; the preview and production
hosting steps consume that same workspace artifact. `delivery.tfplan.txt` is
the human-review evidence for the Terraform plan used by the build.

## Preview

Run the build with the reviewed project and bucket substitutions. The default
`_PROMOTE=false` creates a Cloud Run revision with zero traffic and a Firebase
preview channel, but never changes production traffic. Save the generated
`delivery-manifest.json`, Terraform plan, preview URL, and evaluation result
with the release decision.

The manifest is valid only when `api_image` ends in `@api_digest`; a mutable
tag is not a promotable artifact. Do not rebuild between preview and promotion.

## Promotion

After the sponsor records the release decision and the evaluation gates pass,
rerun the same reviewed build with `_PROMOTE=true`,
`_RELEASE_APPROVED=true`, and a non-empty `_RELEASE_DECISION` identifier. The
promotion step first validates the saved manifest, then routes 100% of Cloud
Run traffic to the exact preview revision and deploys the already-built
frontend artifact to the production Firebase Hosting site.

There is no automatic production promotion. An empty or false release decision
leaves production unchanged.

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
