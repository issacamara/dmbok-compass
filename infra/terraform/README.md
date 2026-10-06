# DMBOK Compass GCP foundation

This directory is the infrastructure source of truth for the first-release
GCP foundation. It provisions the regional managed services, Firebase project
and email/password Authentication, deny-by-default Firestore rules, the corpus
bucket, Artifact Registry, Firebase Hosting metadata, Secret Manager metadata,
and separate service identities for runtime, worker, scheduler, build, and
hosting concerns.

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

Secret values must be created out of band with `gcloud secrets versions add` or
the approved secret-management workflow. This configuration creates only
Secret Manager metadata, so secret values never enter Terraform configuration
or state.

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

From this directory:

```sh
terraform init
terraform fmt -check -recursive
terraform validate
terraform plan \
  -var='project_id=YOUR_PROJECT_ID' \
  -var='corpus_bucket_name=globally-unique-bucket-name'
```

Applying requires explicit project authorization and a reviewed plan:

```sh
terraform apply \
  -var='project_id=YOUR_PROJECT_ID' \
  -var='corpus_bucket_name=globally-unique-bucket-name'
```

The GCP project and billing account are intentionally inputs rather than
managed resources. This avoids accidental project creation and keeps project
selection an explicit release decision.
