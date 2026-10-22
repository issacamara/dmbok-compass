"""Cloud Run entry point for deterministic PDF corpus ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
from collections.abc import Callable
from threading import Event
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import firestore, storage

from app.contracts import CorpusServingBinding, CorpusVersion
from app.corpus.chunking import SourcePage, chunk_pages
from app.corpus.indexing import VECTOR_SCHEMA_VERSION
from app.corpus.manifest import create_source_manifest, default_derivation
from app.corpus.firestore import FirestoreCorpusVersionStore
from app.ingestion.extract import extract_pdf
from app.ingestion.job import CorpusIngestionJob, IngestionInterrupted
from app.rag import LangChainRagFactory, LangChainRagSettings
from app.settings import RuntimeConfiguration


def main() -> None:
    termination_requested = Event()
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: termination_requested.set())
    try:
        _run(termination_requested.is_set)
    except IngestionInterrupted:
        print(json.dumps({"event": "dmbok_ingestion_interrupted", "exit_code": 1}))
        raise SystemExit(1) from None
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _run(termination_requested: Callable[[], bool]) -> None:
    configuration = RuntimeConfiguration.from_environment()
    project = configuration.runtime.project
    runtime = configuration.runtime
    settings = LangChainRagSettings(
        project=runtime.project,
        location=runtime.location,
        vector_collection=configuration.vector_collection,
        answer_provider=runtime.provider,
        primary_model=runtime.primary_model,
        fallback_model=runtime.fallback_model,
        json_schema_supported=configuration.json_schema_supported,
        output_token_cap=configuration.output_token_cap,
    )
    client = firestore.Client(project=project)
    versions = FirestoreCorpusVersionStore(client, vector_collection=settings.vector_collection)
    cleanup_version_id = os.environ.get("CORPUS_CLEANUP_VERSION", "").strip()
    if cleanup_version_id:
        result = versions.cleanup_staging(
            cleanup_version_id,
            dry_run=os.environ.get("CORPUS_CLEANUP_APPLY", "").strip().lower() != "true",
        )
        print(
            json.dumps(
                {
                    "event": "dmbok_corpus_cleanup",
                    "version_id": result.version_id,
                    "dry_run": result.dry_run,
                    "targets": result.targets,
                }
            )
        )
        return

    bucket_name = _required("CORPUS_BUCKET_NAME")
    object_name = _required("CORPUS_OBJECT_NAME")
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
            derivation=default_derivation(vector_collection=settings.vector_collection),
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
            serving_binding=CorpusServingBinding(
                corpus_version_id=manifest.corpus_version_id,
                vector_collection=settings.vector_collection,
                vector_schema_version=VECTOR_SCHEMA_VERSION,
                embedding_model="gemini-embedding-001",
                ingestion_fingerprint=manifest.manifest_sha256,
                configuration_fingerprint=manifest.derivation_sha256,
            ),
        )
        factory = LangChainRagFactory(client, settings)
        writer = factory.ingestion_writer(
            source_metadata=manifest.corpus_metadata(),
        )
        result = CorpusIngestionJob(
            versions,
            writer,
        ).run(version, chunks, termination_requested=termination_requested)
        print(f"Activated corpus version {result.version.version_id} with {len(result.chunks)} chunks.")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
