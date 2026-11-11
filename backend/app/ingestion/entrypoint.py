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
from app.contracts.v29 import ParsingArtifact
from app.corpus.chunking import ParentReviewException, build_llamaindex_nodes, segment_children, segment_parents
from app.corpus.firestore import FirestoreCorpusVersionStore, FirestoreParentChunkStore
from app.corpus.indexing import HierarchicalCorpusWriter, VECTOR_SCHEMA_VERSION
from app.corpus.manifest import SourceManifest, create_source_manifest, hierarchical_derivation
from app.corpus.staging import FirestoreCorpusStagingAttemptStore
from app.corpus.structure_map import sections_from_parsing_artifact
from app.ingestion.extract import ExtractionReport, extract_pdf, recover_logical_sections
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

    attempt_id = os.environ.get("CORPUS_STAGING_ATTEMPT_ID", "").strip()
    progress = _staging_progress_callback(client=client, attempt_id=attempt_id) if attempt_id else None
    try:
        if progress is not None:
            progress("running", 0, 0, None, 0)
        bucket_name = _required("CORPUS_BUCKET_NAME")
        object_name = _required("CORPUS_OBJECT_NAME")
        storage_client = storage.Client(project=project)
        bucket = storage_client.bucket(bucket_name)
    except BaseException as error:
        if progress is not None:
            progress(
                "failed", 0, 0, None, 0,
                "job_interrupted" if isinstance(error, IngestionInterrupted) else "job_failed",
            )
        raise
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "corpus.pdf"
        expected_generation = os.environ.get("CORPUS_SOURCE_GENERATION", "").strip()
        try:
            if expected_generation and not expected_generation.isdecimal():
                raise RuntimeError("The pinned corpus source generation is invalid.")
            source_generation = _download_pinned_source(
                bucket, object_name, source_path,
                expected_generation=int(expected_generation) if expected_generation else None,
            )
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            source_manifest = create_source_manifest(
                bucket=bucket_name,
                object_name=object_name,
                generation=source_generation,
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
            parsing_bytes = None
            parsing_artifact_sha256 = None
            parsing_output_sha256 = None
            if os.environ.get("CORPUS_PARSING_ARTIFACT_URI"):
                parsing_bytes, _, parsing_artifact_sha256 = _read_pinned_object(
                    storage_client,
                    "CORPUS_PARSING_ARTIFACT_URI",
                    "CORPUS_PARSING_ARTIFACT_SHA256",
                )
                parsing_output_sha256 = _required("CORPUS_PARSING_OUTPUT_SHA256")
        except BaseException as error:
            if progress is not None:
                progress(
                    "failed", 0, 0, None, 0,
                    "job_interrupted" if isinstance(error, IngestionInterrupted) else "job_failed",
                )
            raise
        try:
            candidate = build_and_stage_hierarchical_candidate(
                versions=versions,
                source_manifest=source_manifest,
                map_bytes=map_bytes,
                map_uri=map_uri,
                expected_map_sha256=map_sha256,
                approval_bytes=approval_bytes,
                approval_uri=approval_uri,
                expected_approval_sha256=approval_sha256,
                parsing_artifact_bytes=parsing_bytes,
                expected_parsing_artifact_sha256=parsing_artifact_sha256,
                expected_parsing_output_sha256=parsing_output_sha256,
                vector_collection=configuration.vector_collection,
                extract=lambda: extract_pdf(source_path, source_uri=source_manifest.object_uri),
                termination_requested=termination_requested,
            )
        except BaseException as error:
            if progress is not None:
                progress(
                    "failed", 0, 0, None, 0,
                    "job_interrupted" if isinstance(error, IngestionInterrupted) else "job_failed",
                )
            raise
        staged = candidate.version
        try:
            settings = LangChainRagSettings(
                project=project,
                location=configuration.runtime.location,
                vector_collection=configuration.vector_collection,
            )
            factory = LangChainRagFactory(client, settings)
            base_writer = factory.ingestion_writer(
                source_metadata=candidate.manifest.corpus_metadata()
            )
            writer = HierarchicalCorpusWriter(
                base_writer.vector_store,
                FirestoreParentChunkStore(client),
                source_metadata=candidate.manifest.corpus_metadata(),
            )
            result = CorpusIngestionJob(versions, writer).run_hierarchical(
                staged,
                candidate.parents,
                candidate.children,
                unresolved_exceptions=candidate.exceptions,
                termination_requested=termination_requested,
                progress=(
                    lambda state, parent_count, child_count: progress(
                        state, parent_count, child_count, staged.version_id, len(candidate.children)
                    )
                ) if progress is not None else None,
            )
            if result.version.status == "validated" and attempt_id:
                _dispatch_candidate_generation(
                    client=client,
                    project=project,
                    configuration=configuration,
                    attempt_id=attempt_id,
                    corpus_version_id=result.version.version_id,
                )
        except BaseException as error:
            if progress is not None:
                progress(
                    "failed", 0, 0, staged.version_id, len(candidate.children),
                    "job_interrupted" if isinstance(error, IngestionInterrupted) else "job_failed",
                )
            raise
        readiness = (
            "blocked_by_oversized_structures"
            if candidate.exceptions
            else "awaiting_readiness_validation"
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


def _dispatch_candidate_generation(
    *, client: Any, project: str, configuration: RuntimeConfiguration,
    attempt_id: str, corpus_version_id: str,
) -> None:
    """Queue candidate preparation only after ingestion stored a validated corpus."""
    from app.evaluation.automatic import candidate_generator_fingerprint
    from app.evaluation.dispatch import (
        CloudRunCandidateGenerationDispatcher,
        CloudRunJobExecutor,
        EvaluationDispatchClaimStore,
    )
    from app.settings import EvaluationStageConfiguration

    generator = EvaluationStageConfiguration.from_environment("GENERATOR")
    fingerprint = candidate_generator_fingerprint(
        generator.route, generator.model, generator.configuration_version
    )
    dispatcher = CloudRunCandidateGenerationDispatcher(
        EvaluationDispatchClaimStore(client),
        CloudRunJobExecutor(
            project=project,
            location=configuration.runtime.location,
            job_name=os.environ.get("EVALUATION_JOB_NAME", "dmbok-compass-evaluation"),
        ),
    )
    submitted = dispatcher.dispatch(
        attempt_id=attempt_id,
        corpus_version_id=corpus_version_id,
        generator_fingerprint=fingerprint,
    )
    print(json.dumps({
        "event": "candidate_generation_dispatch",
        "status": "submitted" if submitted else "already_submitted",
    }))


def _staging_progress_callback(
    *, client: Any, attempt_id: str,
) -> Callable[..., None]:
    """Bind status writes to the artifact tuple already approved on this attempt."""
    store = FirestoreCorpusStagingAttemptStore(client)
    record = store.get(attempt_id=attempt_id)
    if record is None:
        raise RuntimeError("The approved staging attempt is unavailable.")
    source_artifact = record.get("source")
    map_artifact = record.get("approved_map_artifact")
    approval_artifact = record.get("approval_artifact")
    if (
        not isinstance(source_artifact, dict)
        or not isinstance(map_artifact, dict)
        or not isinstance(approval_artifact, dict)
    ):
        raise RuntimeError("The approved staging attempt artifacts are unavailable.")

    last_parent_count = 0
    last_child_count = 0

    def update(
        state: str, parent_count: int, child_count: int,
        corpus_version_id: str | None, chunk_count: int,
        correction_code: str = "job_failed",
    ) -> None:
        nonlocal last_parent_count, last_child_count
        if state not in {"running", "staged", "failed"}:
            raise ValueError("invalid staging progress state")
        if state == "failed":
            parent_count = max(parent_count, last_parent_count)
            child_count = max(child_count, last_child_count)
        else:
            last_parent_count = parent_count
            last_child_count = child_count
        store.record_ingestion_progress(
            attempt_id=attempt_id,
            source=source_artifact,
            map_artifact=map_artifact,
            approval_artifact=approval_artifact,
            state=state,  # type: ignore[arg-type]
            parent_count=parent_count,
            child_count=child_count,
            chunk_count=chunk_count if state == "staged" else None,
            corpus_version_id=corpus_version_id,
            correction_code=correction_code,  # type: ignore[arg-type]
        )

    return update


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
    llamaindex_nodes: tuple[object, ...] = ()


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
    parsing_artifact_bytes: bytes | None = None,
    expected_parsing_artifact_sha256: str | None = None,
    expected_parsing_output_sha256: str | None = None,
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
    if parsing_artifact_bytes is not None:
        if expected_parsing_artifact_sha256 is None or expected_parsing_output_sha256 is None:
            raise RuntimeError("Pinned parsing artifact digests are incomplete.")
        _verify_digest(parsing_artifact_bytes, expected_parsing_artifact_sha256, "parsing artifact")
        try:
            parsing = ParsingArtifact.model_validate_json(parsing_artifact_bytes)
        except ValueError as error:
            raise RuntimeError("Pinned parsing artifact has invalid JSON or schema.") from error
        if parsing.output_sha256 != expected_parsing_output_sha256:
            raise RuntimeError("Pinned parsing output SHA-256 verification failed.")
        try:
            sections = sections_from_parsing_artifact(
                structure_map,
                parsing,
                source_generation=source_manifest.generation,
                source_sha256=source_manifest.content_sha256,
            )
        except ValueError as error:
            raise RuntimeError("Pinned parsing artifact and reviewed structure map are inconsistent.") from error
    else:
        if structure_map.parsing_output_sha256 is not None:
            raise RuntimeError("Reviewed block map requires its pinned parsing artifact.")
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
        parsing_artifact_sha256=expected_parsing_artifact_sha256,
        parsing_output_sha256=expected_parsing_output_sha256,
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
    llamaindex_nodes = build_llamaindex_nodes(segmented.parents, children)
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
        llamaindex_nodes=llamaindex_nodes,
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
        llamaindex_nodes=candidate.llamaindex_nodes,
    )


def _read_pinned_object(client: Any, uri_name: str, digest_name: str) -> tuple[bytes, str, str]:
    uri = _required(uri_name)
    expected_digest = _required(digest_name)
    bucket_name, object_name, generation = _parse_gcs_uri(uri)
    payload = client.bucket(bucket_name).blob(object_name, generation=generation).download_as_bytes()
    _verify_digest(payload, expected_digest, uri_name)
    return payload, uri, expected_digest


def _download_pinned_source(
    bucket: Any, object_name: str, destination: Path, *, expected_generation: int | None = None
) -> int:
    metadata_blob = bucket.blob(object_name)
    if not metadata_blob.exists():
        raise RuntimeError(f"Approved corpus object does not exist: gs://{bucket.name}/{object_name}")
    metadata_blob.reload()
    if metadata_blob.generation is None:
        raise RuntimeError("The corpus object has no immutable generation.")
    generation = int(metadata_blob.generation)
    if expected_generation is not None and generation != expected_generation:
        raise RuntimeError("The corpus source generation does not match its approval.")
    bucket.blob(object_name, generation=generation).download_to_filename(destination)
    return generation


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
