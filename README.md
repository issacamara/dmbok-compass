# DMBOK Compass

DMBOK Compass is a grounded data-management question-answering workspace. The repository contains independent backend and web workspaces so each can be developed and tested locally.

## Local development

### Backend API

From `backend/`, create a virtual environment and install the test extra:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; its readiness endpoint is `GET /health`.

### Web SPA

From `web/`, install dependencies and run the checks or development server:

```bash
npm install
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

Pull requests and pushes to `main` run `.github/workflows/quality.yml`. The workflow
uses read-only repository access and checks the two workspaces independently:

- Backend: install the test extra, run `pytest`, and audit Python dependencies with
  `pip-audit`.
- Web: run `npm ci`, `npm test -- --run`, `npm run build`, and
  `npm audit --audit-level=high`.
- Terraform: run formatting and configuration validation for
  `infra/terraform/`.

Run the same checks locally before opening a pull request:

```bash
cd backend
python -m pip install -e '.[test]' pip-audit 'setuptools>=83.0.0'
pytest
python -m pip uninstall --yes dmbok-compass-backend
pip-audit --strict

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

## GitHub Actions delivery

`.github/workflows/delivery.yml` replaces Cloud Build for immutable preview and
production delivery. It authenticates to GCP with GitHub OIDC and Terraform's
repository-restricted Workload Identity Federation provider; no long-lived
service-account key is stored in GitHub.

Create GitHub environments named `development` and `production`. Configure
these variables separately in each environment from the matching Terraform
outputs and reviewed project settings: `GCP_PROJECT_ID`,
`CORPUS_BUCKET_NAME`, `GCP_WORKLOAD_IDENTITY_PROVIDER`, and
`GCP_BUILD_SERVICE_ACCOUNT`. Set `GCP_PROJECT_ID` to `dev-dmbok-compass` in
development and `prod-dmbok-compass` in production. Protect the `production`
environment with required reviewers.

Pushes to `main` target development by default. Manual runs can select either
environment; a run with `promote=true` and a release decision identifier
promotes the exact reviewed artifacts in the selected project.

## Workspace boundaries

- `backend/` owns the FastAPI API and Python tests.
- `web/` owns the React/Vite/TypeScript single-page application and frontend tests.
- The backend and SPA communicate through HTTP contracts; neither workspace imports the other.
