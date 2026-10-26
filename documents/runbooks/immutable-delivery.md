# Development and production delivery

## Normal path

1. Merge work into the protected `development` branch through a pull request. The `CI` workflow checks backend tests and dependency audit, final API image smoke test, frontend tests/build/audit, Terraform, and LangChain compatibility.
2. The development workflow builds the backend image once, tags it with the commit SHA, and deploys its digest to the development Cloud Run API and jobs. It builds the frontend with development Firebase client settings and deploys Firebase Hosting.
3. Merge the validated work into the protected `main` branch through a pull request. The production workflow searches development history for a commit whose `backend/` tree matches the production commit. If none exists, first merge and deploy that backend source on `development`.
4. GitHub's `production` environment holds the required reviewer and wait timer gate. After approval, the workflow copies the source image by digest from the development Artifact Registry to the production project. It checks that the destination digest matches before deploying the API and jobs. It builds the frontend with production Firebase client settings and deploys production Firebase Hosting.
5. Each run uploads `deployment.json` with source commit, deployment commit, project, and image digest. Both SHAs are also set on Cloud Run as `SOURCE_COMMIT_SHA` and `DEPLOYMENT_COMMIT_SHA`.

A hotfix branched from `main` must also be merged into `development` and built there. A production deployment with a backend tree that has not been built on development stops before the approval gate.

## Prerequisites

- Apply the Terraform IAM changes in both projects. Development WIF accepts `development`; production WIF accepts `main`. The production build service account can read the development Artifact Registry repository.
- GitHub environment variables for each project: `GCP_PROJECT_DEVELOPMENT` or `GCP_PROJECT_PRODUCTION`, `WIF_PROVIDER`, `GCP_SA_DEVELOPMENT` or `GCP_SA_PRODUCTION`, `CORPUS_VERSION_ID`, `RUNTIME_SECRET_VERSION`, `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, and `VITE_FIREBASE_APP_ID`.
- Secret Manager in each project has the selected `answer-provider-api-key` version. GitHub stores no service account JSON key or runtime API key.
- Protect both branches with pull requests; require CI. Configure required reviewers and a wait timer on the `production` GitHub environment.

## Rollback

Record the current Cloud Run API revision, job image digests, Firebase Hosting release, and corpus version before rollback. Route the API back to its previous revision with `gcloud run services update-traffic`, restore the prior Firebase Hosting release, and redeploy the prior image digest to the ingestion/evaluation jobs. If corpus state changed, restore its prior active version as described in the corpus recovery procedure. Verify health, citations, and the frontend before accepting the rollback. The previous deployment artifacts identify the source and deployment commits and image digest; do not rebuild the backend for rollback.

The earlier manual release-evidence bundle and zero-traffic preview workflow is retired by this branch-driven process. The release-evidence storage and verification scripts remain in the repository for historical records and manual investigations.
