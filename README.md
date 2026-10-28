# DMBOK Compass

DMBOK Compass is a grounded data-management question-answering workspace. The repository contains independent backend and web workspaces so each can be developed and tested locally.

## Local development

### Production-shaped development stack

Copy `.env.example` to `.env`, fill in the Firebase client settings and
backend provider credentials, then run:

```bash
docker compose up --build
```

The web app is available at `http://localhost:8080` and the API health check at
`http://localhost:8000/health`. The frontend container proxies `/api/*` to the
same backend container boundary used by Firebase Hosting and Cloud Run. The
Compose stack does not contain credentials; Google Application Default
Credentials must be available to the container when Firestore or Vertex AI is
used.

This setup keeps development and production aligned on container images,
runtime configuration names, health checks, port binding, and frontend-to-API
routing. The direct workspace commands below remain useful for fast tests and
frontend iteration.

### Backend API

From `backend/`, create a virtual environment and install the test extra:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
pip install --no-deps -e '.[test]'
pytest
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; its readiness endpoint is `GET /health`.

### Web SPA

From `web/`, install dependencies and run the checks or development server:

```bash
npm ci
npm test
npm run build
npm run dev
```

The Vite development server prints its local URL, normally `http://localhost:5173`.
For Firebase email sign-in, copy `web/.env.example` to `web/.env.local` and fill
in the Firebase web app configuration. Set
`VITE_FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099` to use the local Authentication
emulator during development.

The backend uses Application Default Credentials to verify Firebase ID tokens
and access Firestore. Local development needs Google Application Default
Credentials and the Firebase project ID used by the web app. Registration
creates a pending `users/{Firebase UID}` profile; only an approved, email
verified profile can pass the backend access gate. The initial administrator
must be provisioned out of band in Firestore with `role: "admin"`,
`approval_state: "approved"`, and `email_verified: true` after verifying the
Firebase UID and email.

Every API request that reads a profile first verifies the Firebase ID token and
loads `users/{uid}` from Firestore. Application operations must use the approved
user dependency; administrative operations additionally require `role: "admin"`.
Missing or invalid tokens return HTTP 401, while unverified, pending, rejected,
deactivated, and non-admin profiles return HTTP 403. Registration status remains
available to an authenticated user so the client can explain why access is not
yet available.

## Pull-request quality checks

Every pull request runs `.github/workflows/ci.yml`. The workflow
uses read-only repository access and checks the two workspaces independently:

- Backend: install the test extra, run `pytest`, and audit Python dependencies with
  `pip-audit`.
- Python lint: run pinned Ruff checks.
- Web: run `npm ci`, `npm test -- --run`, `npm run build`, and
  `npm audit --audit-level=high`.
- Terraform: run formatting and configuration validation for
  `infra/terraform/`.
- Compatibility: build the Python 3.11 LangChain contract image.

Run the same checks locally before opening a pull request:

```bash
cd backend
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e '.[test]'
python -m pip install --no-deps pip-audit==2.10.1 setuptools==83.0.0
pytest
python -m pip uninstall --yes dmbok-compass-backend
sed '/^--find-links /d; /^langchain-google-firestore==/,/^    # via /d' requirements.lock > /tmp/backend-requirements-audit.lock
pip-audit --strict --no-deps -r /tmp/backend-requirements-audit.lock

cd ../web
npm ci
npm test -- --run
npm run build
npm audit --audit-level=high

cd ../infra/terraform
terraform fmt -check -recursive .
terraform init -backend=false -input=false
terraform validate
```

## GitHub Actions delivery and resource provisioning

`.github/workflows/cd-development.yml` deploys on pushes to `development`.
It builds the API image once, tags it with the commit SHA, and deploys its
registry digest to Cloud Run. `.github/workflows/cd-production.yml` runs on
pushes to `main` behind the protected `production` environment. It finds a
development commit with the same backend tree, copies that exact image digest
to the production GCP project, verifies the copied digest, and deploys it.
If the backend trees differ, production deployment stops until the change has
been built on `development`. Hotfixes must be merged into both branches.

Both workflows use Workload Identity Federation with `id-token: write`.
Configure the existing GitHub environment **variables**
`GCP_WORKLOAD_IDENTITY_PROVIDER` in each environment. The development environment uses
`GCP_PROJECT_ID` and `GCP_BUILD_SERVICE_ACCOUNT`; production uses
`GCP_PROJECT_PRODUCTION` and `GCP_SA_PRODUCTION`.
they are identifiers, so GitHub Actions secrets are unnecessary. Also set
`CONFIGURATION_VERSION_ID`, `RUNTIME_SECRET_VERSION`, and the four
`VITE_FIREBASE_*` client variables in both environments. After authentication,
the workflows resolve the active ready corpus from the target project's
Firestore `corpus_state/active` publication; do not configure a corpus ID manually.
Set the project IDs
to `dev-dmbok-compass` and `prod-dmbok-compass` respectively. The secret
value itself remains in each project's Secret Manager. `ARTIFACT_ID` is optional
for the bootstrap deployment; after the first approved release, set it to that
release manifest's `candidate_digest` before publishing the release.

Protect `development` and `main` as PR-only branches. Configure the
`production` GitHub environment with required reviewers and a wait timer.
Apply the updated Terraform IAM configuration in both GCP projects before
enabling the workflows: it limits development WIF to `development`, production
WIF to `main`, and lets the production build identity read only the
development image repository.

`.github/workflows/terraform.yml` (`terraform`) plans infrastructure changes for pull requests
targeting `development` or `main`, and applies the reviewed configuration after
those branches receive a merge. Both production planning and production apply
run through the protected `production` environment and require human approval.
Terraform uses separate environment variables
`GCP_BUILD_SERVICE_ACCOUNT` with the matching `GCP_WORKLOAD_IDENTITY_PROVIDER` value.
Pull-request plans use the read-only identity; merged applies use the
privileged identity.

Set these GitHub Environment variables in both environments:

- `GCP_PROJECT_ID` (mapped to `TF_VAR_PROJECT_ID`)
- `GCP_BILLING_ACCOUNT_ID` (mapped to `TF_VAR_BILLING_ACCOUNT_ID`)
- `CORPUS_BUCKET_NAME` (mapped to `TF_VAR_CORPUS_BUCKET_NAME`)
- `HOSTING_SITE` (mapped to `TF_VAR_HOSTING_SITE_ID`)
- `RUNTIME_SECRET_VERSION` (`3`, mapped to `TF_VAR_RUNTIME_SECRET_VERSION`)

The GCS state buckets must exist before the first workflow run:
`dev-dmbok-compass-tfstate` and `prod-dmbok-compass-tfstate`. They are separate
state domains and use the backend files under `infra/terraform/backends/`.
Create them through the approved GCP bootstrap process with uniform bucket-level
access, public access prevention, object versioning, and access limited to the
matching Terraform plan/apply service accounts. The main Terraform configuration
cannot create its own backend bucket before `terraform init`; the first
bootstrap must grant the initial state access, after which Terraform maintains
the bucket IAM bindings.

Firebase Hosting uses the environment project IDs as the site IDs, so the
stable frontend URLs are:

- Development: `https://dev-dmbok-compass.web.app/`
- Production: `https://prod-dmbok-compass.web.app/`

The frontend is built separately for each environment because Firebase client
configuration is embedded during its build. Each deployment records the image
source SHA and deployment SHA in a workflow artifact and Cloud Run runtime
environment. See `documents/runbooks/immutable-delivery.md` for rollback.

## Workspace boundaries

- `backend/` owns the FastAPI API and Python tests.
- `web/` owns the React/Vite/TypeScript single-page application and frontend tests.
- The backend and SPA communicate through HTTP contracts; neither workspace imports the other.
