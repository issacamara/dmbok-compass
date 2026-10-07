# Firestore restore and corpus recovery runbook

This runbook is the recovery evidence for BPR-006 and NFR-010. It covers the
durable application records and the separately reproducible corpus index. It
does not create a multi-region failover environment.

## Recovery objective and backup policy

- Firestore uses the managed daily backup schedule defined in
  `infra/terraform/backup.tf`.
- The schedule retains `2592000s` (30 days) of backups. With one scheduled
  backup per day, this is capped at 30 scheduled copies.
- The recovery objective is to restore user profiles, application
  configuration, aggregate usage, evaluation datasets/runs/results, and the
  release-decision records within 24 hours.
- Production questions, answers, retrieved passages, and request traces are
  not expected in the backup because the application does not persist them.

## Restore drill

Run this exercise against a disposable project or isolated Firestore database;
never overwrite the production database during a drill.

1. Record the drill ID, operator, source backup name, source project, target
   project, and UTC start time. Confirm that the selected backup is no older
   than the 30-day retention window.
2. Restore the Firestore backup into the isolated target using the approved
   Google Cloud restore procedure. Record the restore operation ID and the
   completion timestamp.
3. Verify representative records for `users`, quota/configuration documents,
   aggregate metrics, evaluation datasets/items, evaluation runs/results, and
   release decisions. Confirm that no question, answer, passage, or trace
   content is present in the restored durable records.
4. Rebuild the corpus index from the versioned Cloud Storage PDF and manifest:
   extract the PDF, preserve page and section provenance, derive deterministic
   chunk IDs, generate 768-dimensional embeddings, and write the chunks for a
   new staged `CorpusVersion`.
5. Run the staged corpus completeness checks. The version must be non-empty,
   contain unique chunk IDs, bind every chunk to the staged version, and have
   a 768-dimensional embedding for every chunk. Do not activate an invalid
   version.
6. Activate the validated rebuilt version and verify retrieval filters on the
   active version. If the rebuilt version is unsuitable, use the corpus
   version rollback operation to restore the prior valid active pointer.
7. Record the UTC completion time, restored record counts, rebuilt chunk count,
   active version ID, rollback result (if exercised), and any exceptions.

## Evidence template

| Field | Evidence |
| --- | --- |
| Drill ID / operator | |
| Source backup and age | |
| Restore operation ID | |
| Restore start / completion UTC | |
| Durable record checks | profiles / config / aggregate usage / evaluations / decisions |
| Production interaction content absent | yes / no |
| Source object generation and SHA-256 | |
| Rebuilt corpus version and chunk count | |
| Completeness validation | pass / fail |
| Active pointer verified | pass / fail |
| Rollback verified | pass / fail / not exercised |
| Total elapsed time | |
| Within 24-hour objective | pass / fail |
| Exceptions and follow-up ticket | |

The release approver retains the completed evidence with the recovery review.