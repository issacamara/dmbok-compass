# Evaluator-output retention decision

**Status:** approved for evaluator-only implementation; production interaction
retention remains prohibited  
**Decision date:** 2026-10-21  
**Decision owner:** sponsor / release approver (issacamara)  
**Architecture basis:** A-21, [Architecture §7.3](../ARCHITECTURE.md#73-retention-and-recovery)

## Decision

Candidate outputs generated from administrator-authored evaluation questions may
be stored in an admin-restricted Firestore repository so an authorized reviewer
can assess the exact output produced by a specific evaluation run. This is a
separate evaluator workflow; it does not authorize persistence of production
questions, prompts, retrieved passages, answers, traces, or model outputs.

Raw evaluator output is deleted when its run is finalized or 30 days after
generation, whichever comes first. The 30-day period is a maximum safety TTL
for abandoned or incomplete runs, not a required review period. Finalization
must therefore trigger deletion immediately, and expiry cleanup must be
idempotent and safe to retry.

After raw output is deleted, retain only the minimum audit evidence needed to
reproduce the review decision:

- output digest;
- evaluation run and item identifiers;
- candidate identity and scorer/policy versions;
- assessment and assessment status;
- reviewer actor and review timestamp; and
- this policy version or decision identifier.

The retained records must not contain the raw answer, prompt, retrieved text,
citation excerpts, or other interaction content.

## Access boundary

The output repository is administrator/evaluator-only. Authorization must be
checked at the application boundary for reads, writes, finalization, and
deletion. A valid evaluation item or run identifier alone is not authorization.
Production answering must not call this repository, and a production request
must never be accepted as an evaluator output source.

## Backup and recovery treatment

Firestore backups remain subject to the same privacy boundary and must not be
used to extend raw-output retention beyond the finalization/30-day rule. A
restore may recover the evaluator metadata required for audit, but restored raw
outputs must be expired or deleted before the repository is returned to service
if their approved retention window has elapsed. Recovery evidence must record
the policy check and the deletion result.

The output digest and assessment may survive raw-output deletion and backup
retention for audit, subject to the project’s broader evaluator-record
retention and access controls. This decision does not change the separate
30-copy daily backup limit or 24-hour recovery objective.

## Fixture-policy configuration

The following fixture is the implementation contract for #177. It contains no
production content and is intentionally expressed in policy terms rather than
as a live Firestore configuration:

```yaml
policy_id: evaluator-output-retention-v1
source:
  allowed: administrator_authored_evaluation_question
  production_interactions: reject
storage:
  repository: admin_restricted_firestore
  raw_output: exact_candidate_response
  immutable_key: [evaluation_run_id, evaluation_item_id, candidate_id]
  access: approved_evaluator_or_admin
retention:
  delete_on_run_finalization: true
  max_raw_output_age_days: 30
  retain_after_raw_deletion:
    - output_digest
    - evaluation_run_id
    - evaluation_item_id
    - candidate_identity
    - scorer_version
    - assessment
    - assessment_status
    - reviewer_actor
    - reviewed_at
    - policy_id
recovery:
  raw_output_must_not_outlive_policy: true
  restored_expired_raw_output: delete_before_service
```

## Negative access and deletion examples

Persistence implementations must reject these cases without storing or
returning raw content:

| Case | Required result |
| --- | --- |
| Unauthenticated or non-admin read | Deny access; do not reveal whether output exists. |
| Authenticated user read without evaluator/admin role | Deny access; do not reveal raw output or metadata beyond the safe error. |
| Write sourced from a production question or request | Reject at the boundary; no document is created. |
| Write with a missing, unknown, or stale run/item/candidate binding | Reject; no document is created. |
| Finalization by an unauthorized actor | Deny the operation; raw output remains governed by its existing policy window. |
| Finalization by an authorized actor | Delete raw output and preserve only the approved digest/assessment audit record. |
| Retry of an already-completed deletion | Succeed idempotently or report already deleted; never recreate raw output. |
| Read after finalization or expired TTL | Return no raw output; retained audit metadata may remain available to an authorized reviewer. |
| Restore of raw output past its policy window | Delete it before service is restored; record the cleanup evidence. |

## Release gate and contract impact

This decision is the prerequisite for implementing evaluator-only output
persistence in #177. A missing or invalid policy identifier must fail closed.
Implementations must add boundary tests for authorization, production-source
rejection, immutability, finalization deletion, 30-day expiry, idempotent
cleanup, restart/recovery, and the retained digest/assessment shape.

The decision consumes the v2.1 evaluation contracts for an authored evaluation
run/item, candidate identity, exact output digest, assessment, and reviewer
audit metadata. It exposes the `evaluator-output-retention-v1` policy and its
deletion/access rules; it introduces no runtime API or shared-schema change.
