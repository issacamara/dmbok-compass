# Final release evidence bundle

The evidence bundle is the promotion record for one immutable release. It
must be assembled from the exact Cloud Build commit, API image digest,
frontend digest, preview revision, corpus version, evaluation dataset, model,
and configuration that were reviewed. The bundle is evidence, not a second
deployment mechanism: promotion still consumes the reviewed manifest described
in [the immutable delivery runbook](../runbooks/immutable-delivery.md).

## Required evidence

`release-evidence.example.json` is a copyable structure. Replace every
example identifier with an immutable artifact or a dated run record; do not
use screenshots without a stable URL, object name, run ID, or document ID.

The validator requires passing evidence for:

- the artifact manifest and zero-traffic preview revision;
- the completed evaluation run, bound to dataset, corpus, configuration, and
  model versions, with all six metrics at the thresholds in the BRD;
- provider availability and reviewed upstream privacy/data terms;
- privacy/non-retention verification;
- desktop/mobile accessibility and keyboard checks;
- daily backup and a restore exercise within 24 hours;
- three-user response-time evidence (95% within 15 seconds);
- the €5 infrastructure forecast and provider budget controls;
- rollback evidence for the API, frontend, and corpus pointer; and
- publisher-rights acceptance plus the sponsor's release decision.

An absent, failed, or stale record blocks promotion. An exception is only
acceptable when it includes the approver, rationale, and ISO-8601 approval
time; the validator reports it so the sponsor's accepted risk remains
distinguishable from a clean gate.

## Verification

Run the deterministic check from the repository root after setting the
workspace Python path used by the backend:

```sh
PYTHONPATH=backend python scripts/verify_release_evidence.py release-evidence.json
```

Store the validated JSON, `delivery-manifest.json`, Terraform plan text,
evaluation report, preview URL, provider decision, recovery drill, and release
decision together. Do not put questions, answers, retrieved passages, or
request traces in this bundle; only evaluator-authored test records and
content-free operational evidence belong here.
