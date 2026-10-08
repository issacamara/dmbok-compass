"""Cloud Run entry point for deterministic PDF corpus ingestion."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import firestore, storage

from app.contracts import CorpusVersion
from app.corpus.chunking import SourcePage, chunk_pages
from app.corpus.embedding import create_vertex_document_embedder
from app.corpus.indexing import CorpusEmbeddingWriter, FirestoreChunkRepository
from app.corpus.manifest import create_source_manifest
from app.corpus.firestore import FirestoreCorpusVersionStore
from app.ingestion.extract import extract_pdf
from app.ingestion.job import CorpusIngestionJob


def main() -> None:
    project = _required("GOOGLE_CLOUD_PROJECT")
    bucket_name = _required("CORPUS_BUCKET_NAME")
    object_name = _required("CORPUS_OBJECT_NAME")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1")
    bucket = storage.Client(project=project).bucket(bucket_name)
    blob = bucket.blob(object_name)
    if not blob.exists():
        raise RuntimeError(f"Approved corpus object does not exist: gs://{bucket_name}/{object_name}")
    blob.reload()
    if blob.generation is None:
        raise RuntimeError("The corpus object has no immutable generation.")
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "corpus.pdf"
        blob.download_to_filename(source_path)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        manifest = create_source_manifest(
            bucket=bucket_name,
            object_name=object_name,
            generation=int(blob.generation),
            content_sha256=digest,
        )
        report = extract_pdf(source_path, source_uri=manifest.object_uri)
        pages = tuple(
            SourcePage(page=item.page, section=item.section or "Unclassified", text=item.text)
            for item in report.pages
            if item.text
        )
        chunks = chunk_pages(pages, corpus_version_id=manifest.corpus_version_id)
        if not chunks:
            raise RuntimeError("The corpus PDF produced no searchable chunks.")
        version = CorpusVersion(
            version_id=manifest.corpus_version_id,
            source_uri=manifest.object_uri,
            embedding_model="gemini-embedding-001",
            status="staged",
            created_at=datetime.now(timezone.utc),
        )
        client = firestore.Client(project=project)
        writer = CorpusEmbeddingWriter(
            create_vertex_document_embedder(project=project, location=location),
            FirestoreChunkRepository(client),
        )
        result = CorpusIngestionJob(FirestoreCorpusVersionStore(client), writer).run(version, chunks)
        print(f"Activated corpus version {result.version.version_id} with {len(result.chunks)} chunks.")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
