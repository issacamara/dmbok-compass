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

The rollback test uses synthetic current/previous manifests and the verified development image digest. It verifies that the forward plan preserves the exact API/admin image, the rollback plan restores the prior image and serving binding, and both protected-gate and owner approvals are required. The recovery verifier tests use isolated synthetic evidence to check the 24-hour restore bound, 30-backup limit, immutable source derivation, and serving-binding comparison. The reference-aware retention tests cover source, parse, map, approval, dataset, and review artifacts; active or rollback references protect them, missing recovery paths are refused, and concurrent reference changes abort the inventory.

### Nonproduction rehearsal evidence (2026-11-13)

- Firestore recovery was rehearsed in development by restoring the Nov 13 `(default)` database backup into the isolated database `issue372-recovery-drill-20261113`. The restored database reached `Available/Ready` in approximately 75 minutes, within the 24-hour recovery objective. The console showed 30 retained backup copies with 30-day retention. The backup copy identifier and restore operation identifier were not recorded in the evidence available for this run.
- A bounded, offline rebuild preview used the saved parsing artifact and reviewed structure map, with the repository-pinned `tiktoken==0.12.0` `cl100k_base` encoding. It verified the source generation `1794533783986389` and source SHA-256 `4d14b28ac6adb71b9551e66e109666fa6b467d525fd6d57704cb88dd9c8d76b4`; the parsing output SHA-256 was `eb02313392ccf07e941e7546e9107e8601fb6ee426af251f81cb3438472e6183`. The structure map content hash was `8b4b962df81a1fa9aa73d2df22989e76621c32baa54314ac5447ee674c293086` (map ID `staged-4d14b28ac6ad-8b4b962df81a`). The preview covered physical pages 21, 25, 58, 78, and 105, producing 6 sections, 7 parent chunks, and 15 child chunks. It completed locally with no cloud writes and no active corpus pointer change.
- Saved-object metadata was checked without exposing document content: source generation `1794533783986389` (MD5 `vdD8eVu8q7X4UJSgzQLERA==`, CRC32C `k/fPnQ==`); parsing artifact generation `1794558805916464` (MD5 `Cp6PxLuPdLzDeKcmo8hrNw==`, CRC32C `Nh/+vQ==`); and reviewed map revision 4 generation `1794560824292716` (MD5 `eoUK5zDpxMX8QXXcqxYw4Q==`, CRC32C `LMNanQ==`). The parse artifact binds to the recorded source generation and digest, and the map binds to the parse-output digest. An approval record was present at generation `1794561604332173`.
- The database restore and offline segmentation preview are separate recovery checks; this rehearsal did not restore the corpus serving pointer, perform embedding writes, run a serving evaluation against the restored database, or activate production. Production approval and deployment remain untouched.

Retain each rollback input until the promoted release has passed verification and a fresh reference snapshot confirms that no active or rollback release still uses it. The cleanup inventory remains dry-run only; any later deletion requires a separate approved operation. The repository has no verified merge queue, so the epic integrator serializes merges after rechecking the PR head, current `development` base, review, and required checks. Do not run parallel final merges.
