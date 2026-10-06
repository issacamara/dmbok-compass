# PDF provenance extraction spike

## Decision

The spike uses `pypdf` because it is a small, pure-Python parser that can
extract text from searchable PDFs without introducing a native system
dependency. The extractor records its parser version in every report so a
future parser upgrade is visible in reproducibility evidence.

`backend/app/ingestion/extract.py` emits one-based page records, a SHA-256 of
the exact source bytes, the supplied immutable source URI, the recovered
chapter or numbered section, and the normalized page text. A heading carries
forward to later pages until another heading is found. Pages with no text or
no recoverable heading are retained and reported as failures; they are never
silently dropped.

## Human review procedure

For the approved source PDF, inspect a representative sample containing the
first page, every chapter opening, section transitions, pages with headers or
footers, and the final page. Compare the report's page and section against the
rendered PDF and record any mislabels or extraction failures. The source URI,
SHA-256, parser version, report, and reviewer decision belong with the versioned
Cloud Storage ingestion artifacts.

The synthetic fixture verifies the mechanics and repeatability of page and
section recovery. It does not substitute for human review of the copyrighted
source PDF; that review remains a release gate under A-07 and RISK-07.

### Reproducible fixture review

| Fixture page | Expected visible heading | Extracted citation section | Decision |
| --- | --- | --- | --- |
| 1 | `1.1 Purpose` | `1.1 Purpose` | Pass; no mislabel |
| 2 | `1.2 Roles` | `1.2 Roles` | Pass; no mislabel |

The fixture review is reproducible through `backend/tests/test_extract.py`.
The approved DMBOK PDF was not available in this checkout, so no claim is made
about its layout-specific accuracy until the operator runs the human-review
procedure above against the versioned source object.

## Known failure cases

- Image-only or otherwise unsearchable pages produce a `no searchable text`
  failure and cannot safely receive a citation.
- Pages without a chapter or numbered heading inherit the last known section;
  if none exists, the report records a missing-section failure.
- Complex layouts, footnotes, and running headers may alter reading order;
  these require the human-review sample before the corpus can be promoted.
