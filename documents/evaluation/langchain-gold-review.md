# Candidate Gold-Annotation Review

Run the bounded remapping maintenance after every candidate corpus rebuild. It
does not write Firestore, activate a dataset, or approve an annotation.

```sh
PYTHONPATH=backend python scripts/remap_gold_annotations.py \
  --input /secure/operator/candidate-remap.json \
  --output /secure/operator/candidate-gold-review.json
```

The input is an object with these fields:

- `binding`: `corpus_version_id`, `candidate_snapshot_sha256`, and
  `source_provenance_sha256`. All values identify the same frozen candidate.
- `dataset`: the evaluator-authored `EvaluationDataset` being reconsidered.
- `candidate_chunks`: candidate `DocumentChunk` objects. Each may include a
  `section_origin` of `explicit` or `carried`.

The output contains only `candidate` review records. An exact unique supporting
passage match proposes one candidate chunk ID together with one-based page and
section provenance. It also flags carried headings and `Unclassified` sections
for sponsor review. Unknown, ambiguous, missing, and changed support have no
proposed chunk ID, and therefore cannot inherit an earlier approval.

Use the existing administrator import and review flow to create a new candidate
dataset, then compare its binding and this review output before an administrator
explicitly approves a matched annotation. A candidate review stored with a
generation binds every annotation to the candidate corpus, frozen snapshot, and
source provenance; any legacy approval is rejected. Review the authored fixture
across `definitions`, `explanations`, `comparisons`, `study`, and `scenarios`,
including answerable, partial, and unanswerable oracle cases, before selecting
the 30–50 approved release subset.
