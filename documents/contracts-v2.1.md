# v2.1 boundary contracts

Version 2.1 freezes the cross-workstream boundary for candidate evaluation and
grounded answers. The JSON fixture at
`backend/tests/fixtures/architecture-v21.json` is the executable example for
the Python and browser contract tests. Consumers should depend on these
schemas and the protocols in `backend/model/protocol.py`, never on another
workstream's storage or provider implementation.

## Contract ownership

| Contract | Owner | Boundary |
| --- | --- | --- |
| `CandidateIdentity` | release/evaluation | immutable artifact, lock, corpus, configuration, and scorer references |
| `RetrievalResult` | retrieval | ordered candidate-pinned IDs, aligned scores, strategy, and independent stage status |
| `EvaluationCase` / `RunItemResult` | evaluation | evaluator-authored input and content-free result |
| `AnswerOutcome` / `CitationEvidence` | answering | truthful outcome, safe trace/quota, and bounded source-span association |
| `RuntimeSettings` | runtime | non-secret deployment/model references and optional artifact/secret versions |

## Outcome and compatibility rules

Invalid structured provider output maps to 502, unavailable dependencies to
503, and an exceeded deadline to 504. A completed request that cannot support
an answer is an `evidence_refusal` and remains HTTP 200; it is not an outage.
Trace and quota metadata remain safe and available on every answer outcome.

Version 2.1 consumers accept only the `2.1` fixture schema. A candidate
identity is immutable after a run references it. Retrieval scores remain
positionally aligned with ordered IDs. Runtime settings can carry a secret
version reference, but never a secret value. The optional lexical artifact is
an identity reference only and does not make lexical retrieval a serving
requirement.

## Migration notes

Existing v1.9 `AnswerResponse`, `RetrievedPassage`, and evaluation models remain
available for compatibility. New v2.1 integrations should map them at the
boundary into the frozen models rather than changing the v1.9 shapes in place.
The adapter and repository protocols are deliberately async/fake-friendly so
storage and provider implementations can migrate independently. No migration
should persist production questions, prompts, passages, answers, raw provider
responses, or secret material.
