# Hybrid retrieval adoption experiment

Status: **retain-dense-only**

The optional hybrid/rerank implementation remains disabled. This decision does
not enable serving, tune results, or publish a release.

## Evidence status

The frozen dense candidate evidence in `dense-candidate-evidence.md` is a
controlled fixture boundary, not live approval evidence. It records that
final-image execution, an approved provider budget, and a deployed three-user
cold/normal/fallback probe are missing. It also contains no prespecified,
reviewed terminology/numbered-section held-out slice of at least 40 cases for
a hybrid comparison.

Accordingly, hybrid has not demonstrated the required five-percentage-point
hit@5 improvement without an overall hit@5 loss, and its SM-02, SM-05, SM-06,
memory, cold-start, privacy, and approved-cost gates remain unmeasured. Dense
only therefore remains the candidate baseline.

## Future comparison protocol

Use `scripts/compare_retrieval_strategies.py` with aggregate results from the
same prespecified held-out slice. It rejects an adoption recommendation unless
the slice has at least 40 cases, hybrid gains at least five percentage points
of hit@5 without an overall loss, all required gates pass, and dense and hybrid
retain distinct calibration-policy versions. The script performs no tuning.

An adoption recommendation creates a distinct non-production hybrid candidate;
it cannot reuse dense candidate approval. Dense-only release has no dependency
on this optional experiment.
