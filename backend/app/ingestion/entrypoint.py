"""Cloud Run entry point for deterministic PDF corpus ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import tempfile
from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

from google.cloud import firestore, storage

from app.contracts import ChildDocumentChunk, CorpusServingBinding, CorpusVersion, ParentChunk, StructureMap
from app.corpus.chunking import ParentReviewException, segment_children, segment_parents
from app.corpus.firestore import FirestoreCorpusVersionStore
from app.corpus.indexing import VECTOR_SCHEMA_VERSION
from app.corpus.manifest import SourceManifest, create_source_manifest, hierarchical_derivation
from app.ingestion.extract import ExtractionReport, extract_pdf, recover_logical_sections
from app.ingestion.job import IngestionInterrupted
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
    client = firestore.Client(project=project)
    versions = FirestoreCorpusVersionStore(client, vector_collection=configuration.vector_collection)
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
    storage_client = storage.Client(project=project)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    if not blob.exists():
        raise RuntimeError(f"Approved corpus object does not exist: gs://{bucket_name}/{object_name}")
    blob.reload()
    if blob.generation is None:
        raise RuntimeError("The corpus object has no immutable generation.")
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "corpus.pdf"
        blob.download_to_filename(source_path)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        source_manifest = create_source_manifest(
            bucket=bucket_name,
            object_name=object_name,
            generation=int(blob.generation),
            content_sha256=digest,
        )
        map_bytes, map_uri, map_sha256 = _read_pinned_object(
            storage_client, "CORPUS_STRUCTURE_MAP_URI", "CORPUS_STRUCTURE_MAP_SHA256"
        )
        approval_bytes, approval_uri, approval_sha256 = _read_pinned_object(
            storage_client,
            "CORPUS_STRUCTURE_MAP_APPROVAL_URI",
            "CORPUS_STRUCTURE_MAP_APPROVAL_SHA256",
        )
        candidate = build_and_stage_hierarchical_candidate(
            versions=versions,
            source_manifest=source_manifest,
            map_bytes=map_bytes,
            map_uri=map_uri,
            expected_map_sha256=map_sha256,
            approval_bytes=approval_bytes,
            approval_uri=approval_uri,
            expected_approval_sha256=approval_sha256,
            vector_collection=configuration.vector_collection,
            extract=lambda: extract_pdf(source_path, source_uri=source_manifest.object_uri),
            termination_requested=termination_requested,
        )
        staged = candidate.version
        readiness = (
            "blocked_by_oversized_structures"
            if candidate.exceptions
            else "awaiting_parent_child_persistence"
        )
        print(
            json.dumps(
                {
                    "event": "dmbok_hierarchical_corpus_staged",
                    "version_id": staged.version_id,
                    "manifest_sha256": candidate.manifest.manifest_sha256,
                    "parent_count": len(candidate.parents),
                    "child_count": len(candidate.children),
                    "review_exception_count": len(candidate.exceptions),
                    "readiness": readiness,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class HierarchicalCorpusCandidate:
    """Verified derivation handed to the parent/child persistence stage."""

    version: CorpusVersion
    manifest: SourceManifest
    structure_map: StructureMap
    parents: tuple[ParentChunk, ...]
    children: tuple[ChildDocumentChunk, ...]
    exceptions: tuple[ParentReviewException, ...]
    ingestion_gate: dict[str, Any]


def build_hierarchical_candidate(
    *,
    source_manifest: SourceManifest,
    map_bytes: bytes,
    map_uri: str,
    expected_map_sha256: str,
    approval_bytes: bytes,
    approval_uri: str,
    expected_approval_sha256: str,
    vector_collection: str,
    extract: Callable[[], ExtractionReport],
) -> HierarchicalCorpusCandidate:
    """Verify immutable source/map approval and build deterministic parent/child data."""

    map_sha256 = _verify_digest(map_bytes, expected_map_sha256, "structure map")
    approval_sha256 = _verify_digest(approval_bytes, expected_approval_sha256, "structure-map approval")
    try:
        structure_map = StructureMap.model_validate(json.loads(map_bytes))
        approval = json.loads(approval_bytes)
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("Pinned structure-map artifacts have invalid JSON or schema.") from error
    ingestion_gate = _verify_map_approval(
        approval,
        structure_map=structure_map,
        source_manifest=source_manifest,
        map_uri=map_uri,
        map_sha256=map_sha256,
    )
    report = extract()
    if report.source_uri != source_manifest.object_uri or report.source_sha256 != source_manifest.content_sha256:
        raise RuntimeError("Extracted PDF identity does not match its immutable source manifest.")
    sections = recover_logical_sections(report, structure_map)
    map_bucket, map_object, map_generation = _parse_gcs_uri(map_uri)
    approval_bucket, approval_object, approval_generation = _parse_gcs_uri(approval_uri)
    derivation = hierarchical_derivation(
        map_id=structure_map.map_id,
        map_uri=f"gs://{map_bucket}/{map_object}",
        map_generation=map_generation,
        map_file_sha256=map_sha256,
        map_content_sha256=structure_map.content_hash,
        approval_uri=f"gs://{approval_bucket}/{approval_object}",
        approval_generation=approval_generation,
        approval_sha256=approval_sha256,
        ingestion_gate=ingestion_gate,
        vector_collection=vector_collection,
    )
    manifest = create_source_manifest(
        bucket=source_manifest.bucket,
        object_name=source_manifest.object_name,
        generation=source_manifest.generation,
        content_sha256=source_manifest.content_sha256,
        derivation=derivation,
    )
    segmented = segment_parents(sections, corpus_version_id=manifest.corpus_version_id)
    children = segment_children(segmented.parents)
    if not segmented.parents or not children:
        raise RuntimeError("The approved structure map produced no hierarchical chunks.")
    version = CorpusVersion(
        version_id=manifest.corpus_version_id,
        source_uri=manifest.object_uri,
        embedding_model="gemini-embedding-001",
        status="staged",
        created_at=datetime.now(timezone.utc),
        serving_binding=CorpusServingBinding(
            corpus_version_id=manifest.corpus_version_id,
            vector_collection=vector_collection,
            vector_schema_version=VECTOR_SCHEMA_VERSION,
            embedding_model="gemini-embedding-001",
            ingestion_fingerprint=manifest.manifest_sha256,
            configuration_fingerprint=manifest.derivation_sha256,
        ),
    )
    return HierarchicalCorpusCandidate(
        version=version,
        manifest=manifest,
        structure_map=structure_map,
        parents=segmented.parents,
        children=children,
        exceptions=segmented.exceptions,
        ingestion_gate=ingestion_gate,
    )


def build_and_stage_hierarchical_candidate(
    *,
    versions: Any,
    termination_requested: Callable[[], bool] = lambda: False,
    **kwargs: Any,
) -> HierarchicalCorpusCandidate:
    """Verify and derive first, then idempotently persist only the staged identity."""

    candidate = build_hierarchical_candidate(**kwargs)
    if termination_requested():
        raise IngestionInterrupted("termination requested before candidate staging")
    staged_version = versions.stage(candidate.version)
    return candidate.__class__(
        version=staged_version,
        manifest=candidate.manifest,
        structure_map=candidate.structure_map,
        parents=candidate.parents,
        children=candidate.children,
        exceptions=candidate.exceptions,
        ingestion_gate=candidate.ingestion_gate,
    )


def _read_pinned_object(client: Any, uri_name: str, digest_name: str) -> tuple[bytes, str, str]:
    uri = _required(uri_name)
    expected_digest = _required(digest_name)
    bucket_name, object_name, generation = _parse_gcs_uri(uri)
    payload = client.bucket(bucket_name).blob(object_name, generation=generation).download_as_bytes()
    _verify_digest(payload, expected_digest, uri_name)
    return payload, uri, expected_digest


def _parse_gcs_uri(uri: str) -> tuple[str, str, int]:
    if not uri.startswith("gs://") or "#" not in uri:
        raise RuntimeError("Pinned corpus artifacts must use gs:// URIs with immutable generations.")
    bucket_and_object, generation_text = uri[5:].rsplit("#", 1)
    bucket, separator, object_name = bucket_and_object.partition("/")
    if not separator or not bucket or not object_name:
        raise RuntimeError("Pinned corpus artifact URI is invalid.")
    try:
        generation = int(generation_text)
    except ValueError as error:
        raise RuntimeError("Pinned corpus artifact generation is invalid.") from error
    if generation < 1:
        raise RuntimeError("Pinned corpus artifact generation must be positive.")
    return bucket, object_name, generation


def _verify_digest(payload: bytes, expected: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"Pinned {label} SHA-256 must be a lowercase digest.")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Pinned {label} SHA-256 verification failed.")
    return actual


def _verify_map_approval(
    approval: Any,
    *,
    structure_map: StructureMap,
    source_manifest: SourceManifest,
    map_uri: str,
    map_sha256: str,
) -> dict[str, Any]:
    if not isinstance(approval, dict) or approval.get("schema_version") != "corpus-owner-approval-v1":
        raise RuntimeError("Structure-map approval has an unsupported schema.")
    source = approval.get("source")
    approved_map = approval.get("map")
    gate = approval.get("ingestion_gate")
    if approval.get("decision") != "approved" or not isinstance(source, dict) or not isinstance(approved_map, dict):
        raise RuntimeError("Structure map has no valid explicit owner approval.")
    if (
        source.get("uri") != source_manifest.object_uri
        or source.get("generation") != source_manifest.generation
        or source.get("sha256") != source_manifest.content_sha256
    ):
        raise RuntimeError("Structure-map approval does not match the immutable PDF source.")
    map_bucket, map_object, map_generation = _parse_gcs_uri(map_uri)
    expected_artifact_uri = f"gs://{map_bucket}/{map_object}"
    if (
        approved_map.get("artifact_uri") != expected_artifact_uri
        or str(approved_map.get("generation")) != str(map_generation)
        or approved_map.get("file_sha256") != map_sha256
        or approved_map.get("content_hash") != structure_map.content_hash
        or approved_map.get("map_id") != structure_map.map_id
        or approved_map.get("schema_version") != structure_map.schema_version
    ):
        raise RuntimeError("Structure-map approval does not match the pinned map artifact.")
    if not isinstance(gate, dict) or "oversized_indivisible_list_items_or_table_rows" not in gate:
        raise RuntimeError("Structure-map approval is missing its ingestion gate decision.")
    return gate


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
