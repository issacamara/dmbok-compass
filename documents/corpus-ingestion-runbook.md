# Corpus ingestion and recovery

Corpus maintenance is an offline, administrator-only operation.  Each run
binds one approved, versioned PDF source object to a deterministic
`CorpusVersion` and deterministic `DocumentChunk` identifiers.

The ingestion job writes the new version as `staged`, persists all chunks, and
checks that the staged corpus is non-empty, has unique chunk IDs, has matching
version IDs, and contains 768-dimensional embeddings.  Only a complete,
validated version can become `active`.  The active pointer and version status
change together; retrying a completed run is a no-op for the active version.

If validation or embedding fails, the staged version is not eligible for
activation and the existing active pointer remains unchanged.  If a promoted
version must be withdrawn, run the rollback operation.  It restores the prior
valid version's pointer and marks the withdrawn version retired.  Rebuilds are
safe because source generation, corpus version, page, and chunk ordinal form
stable identities.
