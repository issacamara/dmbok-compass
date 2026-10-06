# Corpus extraction fixtures

`searchable-provenance-fixture.pdf.b64` is a tiny, synthetic searchable PDF
fixture. It is base64-encoded so the fixture remains text-only in source
control; tests materialise it before extraction. It contains two pages, a
chapter heading, and numbered section headings.

The real DMBOK PDF is intentionally not checked into this repository. The
operator must run the same extractor against the approved, versioned Cloud
Storage object and retain the generated report as an ingestion artifact.
