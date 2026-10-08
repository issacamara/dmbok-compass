# DMBOK Compass GCP foundation

This directory is the infrastructure source of truth for the first-release
GCP foundation. It provisions the regional managed services, Firebase project
and email/password Authentication, deny-by-default Firestore rules, the corpus
bucket, Artifact Registry, Firebase Hosting metadata, Secret Manager metadata,
and separate service identities for runtime, worker, scheduler, build, and
hosting concerns. GitHub Actions authenticates as the build identity through
the repository-restricted Workload Identity Federation provider.

Apply this foundation separately to the `dev-dmbok-compass` and
`prod-dmbok-compass` projects. Each project has its own Artifact Registry,
Cloud Run resources, Firebase Hosting site, secrets, service identities, and
GitHub OIDC provider. Store the resulting provider and build-service-account
outputs in the matching GitHub `development` or `production` environment.

Terraform state is stored remotely in separate, versioned GCS buckets:

- Development: `gs://dev-dmbok-compass-tfstate/terraform/foundation/default.tfstate`
- Production: `gs://prod-dmbok-compass-tfstate/terraform/foundation/default.tfstate`

Create and protect those buckets in the bootstrap step before initializing the
main configuration. The backend configuration files are
`backends/development.hcl` and `backends/production.hcl`.

The default region is `europe-west1`. The corpus bucket is regional, private,
uniformly access-controlled, versioned, and protected against public access.
Replacing the source object preserves the previous generation for rollback;
objects older than 30 days are lifecycle-managed. Runtime configuration is deliberately
not deployed here yet: Cloud Run services and jobs will consume the identity
outputs when their application modules are implemented. The scheduler identity
also receives no project-wide `run.invoker` grant; that permission belongs on
the specific Cloud Run target.

The application service is intended to start with zero minimum instances, a
maximum of two instances, concurrency eight, one vCPU, and 512 MiB memory.
Worker jobs are bounded batch execution with one task and no always-on
instance. These limits keep the initial deployment aligned with the three-user
capacity target and the monthly infrastructure budget.

`delivery.tf` defines the API service and bounded ingestion/evaluation jobs.
GitHub Actions supplies an immutable registry-digest image during the reviewed
plan; its preview step creates a zero-traffic revision and the production step
routes traffic only after an explicit release decision and protected
`production` environment approval. See
`documents/runbooks/immutable-delivery.md` for the evidence and rollback drill.

Secret values must be created out of band with `gcloud secrets versions add` or
the approved secret-management workflow. This configuration creates only
Secret Manager metadata, so secret values never enter Terraform configuration
or state.

The `answer-provider-api-key` secret supplies `OPENROUTER_API_KEY` to the
backend runtime. Optional `OPENROUTER_PRIMARY_MODEL` and
`OPENROUTER_FALLBACK_MODEL` settings override the defaults documented in
`documents/decisions/provider.md`; do not put either the key or model settings
in the browser bundle.

The web app uses public Firebase client configuration from the Firebase
console. Copy `web/.env.example` to `web/.env.local` and fill in the web app's
API key, auth domain, project ID, and app ID. These client values identify the
Firebase project; they are not service account credentials. Local development
can point Authentication at the emulator with
`VITE_FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099`.

Firestore's `document_chunks` collection stores immutable chunk provenance,
the 768-dimensional `gemini-embedding-001` document vector, and its
`corpus_version_id`. The native vector index combines the corpus-version
filter with the embedding field so retrieval can never cross corpus versions.

## Validate and plan

From this directory, after the matching state bucket exists:

```sh
terraform init \
  -backend-config=backends/development.hcl \
  -reconfigure
terraform fmt -check -recursive
terraform validate
terraform plan \
  -var='project_id=dev-dmbok-compass' \
  -var='corpus_bucket_name=globally-unique-bucket-name'
```

For production, use `backends/production.hcl`, set
`project_id=prod-dmbok-compass`, and use the production corpus bucket name.

Applying requires explicit project authorization and a reviewed plan:

```sh
terraform apply \
  -var='project_id=YOUR_PROJECT_ID' \
  -var='corpus_bucket_name=globally-unique-bucket-name'
```

The GCP project and billing account are intentionally inputs rather than
managed resources. This avoids accidental project creation and keeps project
selection an explicit release decision.
