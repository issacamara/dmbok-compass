# Corpus ingestion and recovery

Corpus maintenance is an offline, administrator-only operation. Each run
loads the approved, versioned PDF with LangChain `PyPDFLoader` in page mode,
splits page text with the configured 700-word/70-word-overlap LangChain
splitter, and derives stable corpus-version and chunk identifiers from the
source generation and RAG derivation fingerprint.

LangChain generates 768-dimensional Vertex document embeddings and writes
documents through `FirestoreVectorStore.add_documents` to the collection set
by `RAG_VECTOR_COLLECTION` (default `document_chunks_v2`). Each vector document
stores text and embedding at the top level, with source identity, corpus
version, chunk ID, page, section, ordinal, content hash, and schema version in
the nested `metadata` map. Re-running the same source overwrites the same
deterministic IDs rather than creating duplicates.

The ingestion job writes a `staged` corpus version, persists every vector, and
then compares the stored IDs, text, hashes, provenance, and embedding dimensions
with the derived manifest. Only a complete version becomes `validated`.
Activation uses a Firestore transaction and compare-and-set against the corpus
that was active when ingestion began, so a stale concurrent job cannot replace
a newer activation. The same transaction retires the old version, activates the
candidate, and updates `corpus_state/active` with current and previous IDs.

If loading, embedding, writing, or validation fails, the active pointer remains
unchanged. To withdraw a promoted version, run the corpus-version store's
`rollback()` maintenance operation in the same runtime image and configuration.
It atomically restores the recorded previous valid version and retires the
withdrawn version. An explicit previously valid version ID may be supplied when
an administrator needs a targeted rollback. Keep old vector documents and the
versioned source object until promotion and rollback evidence is accepted.

Before production promotion, rebuild the full candidate corpus, verify filtered
retrieval from the active version, run the approved evaluation and load gates,
and record a rollback drill. The exposed development OpenRouter key must be
rotated before any further shared-provider validation or deployment.
