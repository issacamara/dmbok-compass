# DMBOK Compass GCP foundation

This directory is the infrastructure source of truth for the first-release
GCP foundation. It provisions the regional managed services, Firestore Native,
the corpus bucket, Artifact Registry, Firebase Hosting metadata, Secret
Manager metadata, and separate service identities for runtime, worker,
scheduler, build, and hosting concerns.

The default region is `europe-west1`. Runtime configuration is deliberately
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
