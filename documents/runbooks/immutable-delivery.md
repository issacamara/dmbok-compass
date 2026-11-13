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
- GitHub environment variables for each project: `GCP_PROJECT_DEVELOPMENT` or `GCP_PROJECT_PRODUCTION`, `WIF_PROVIDER`, `GCP_SA_DEVELOPMENT` or `GCP_SA_PRODUCTION`, `RUNTIME_SECRET_VERSION`, `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, and `VITE_FIREBASE_APP_ID`. After authentication, workflows resolve and validate `corpus_state/active` and its ready `corpus_versions/{version_id}` record; no corpus ID is manually configured.
- Secret Manager in each project has the selected `answer-provider-api-key` version. GitHub stores no service account JSON key or runtime API key.
- Protect both branches with pull requests; require CI. Configure required reviewers and a wait timer on the `production` GitHub environment.

## Rollback

Record the current Cloud Run API revision, job image digests, Firebase Hosting release, and corpus version before rollback. Route the API back to its previous revision with `gcloud run services update-traffic`, restore the prior Firebase Hosting release, and redeploy the prior image digest to the ingestion/evaluation jobs. If corpus state changed, restore its prior active version as described in the corpus recovery procedure. Verify health, citations, and the frontend before accepting the rollback. The previous deployment artifacts identify the source and deployment commits and image digest; do not rebuild the backend for rollback.

The earlier manual release-evidence bundle and zero-traffic preview workflow is retired by this branch-driven process. The release-evidence storage and verification scripts remain in the repository for historical records and manual investigations.

## v2.9 promotion and rollback rehearsal

This rehearsal validates the release path without publishing to production. The development image under review is the immutable image built and deployed for commit `302076ad76709ff30d3298abe76c37c4b6414851` by development run [#96](https://github.com/issacamara/dmbok-compass/actions/runs/43463514198): `europe-west1-docker.pkg.dev/dev-dmbok-compass/dmbok-compass/api@sha256:8bd5f80506146691b2578d5209c200d28ab0f59f96f05a35b39b9184c47844e0`. The production workflow resolves that source digest, copies the image, and fails unless the destination digest matches. The production job remains behind its `production` environment gate; this rehearsal does not request approval, start the production workflow, activate a release, or deploy an image.

Run the repeatable local checks from the repository root:

```bash
cd backend
python -m pytest tests/test_delivery_rollback.py tests/test_recovery.py tests/test_source_retention.py
```

The rollback test uses synthetic current/previous manifests and the verified development image digest. It verifies that the forward plan preserves the exact API/admin image, the rollback plan restores the prior image and serving binding, and both protected-gate and owner approvals are required. The recovery verifier tests use isolated synthetic evidence: they enforce a restore time no greater than 24 hours, no more than 30 retained daily backups, an immutable source generation and derivation, and a rebuild whose serving binding matches the saved evaluation snapshot. The reference-aware retention tests cover the saved source, parse, map, approval, dataset, and review artifacts; active or rollback references protect them, missing recovery paths are refused, and concurrent reference changes abort the inventory. These are preflight and evidence checks, not a live Firestore restore or corpus rebuild.

Retain each rollback input until the promoted release has passed verification and a fresh reference snapshot confirms that no active or rollback release still uses it. The cleanup inventory remains dry-run only; any later deletion requires a separate approved operation. The repository has no verified merge queue, so the epic integrator serializes merges after rechecking the PR head, current `development` base, review, and required checks. Do not run parallel final merges.
