from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from app.contracts import CorpusVersion, StructureMap, StructureMapEntry
from app.corpus import InMemoryCorpusVersionStore, TextChunk, create_source_manifest
from app.ingestion import entrypoint
from app.ingestion.extract import ExtractionReport, PageProvenance


SOURCE_SHA256 = "a" * 64
MAP_URI = "gs://corpus/maps/reviewed.json#12"
APPROVAL_URI = "gs://corpus/approvals/reviewed.json#13"
INGESTION_GATE = {"oversized_indivisible_list_items_or_table_rows": "blocking_if_found"}


def _artifacts(*, map_content_hash: str = "b" * 64) -> tuple[bytes, bytes, str, str]:
    structure_map = StructureMap(
        map_id="reviewed-map",
        source_sha256=SOURCE_SHA256,
        content_hash=map_content_hash,
        entries=[StructureMapEntry(section_path=("Chapter 1",), page_start=1, page_end=2)],
    )
    map_bytes = json.dumps(structure_map.model_dump(mode="json"), sort_keys=True).encode()
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()
    approval = {
        "schema_version": "corpus-owner-approval-v1",
        "decision": "approved",
        "source": {
            "uri": "gs://corpus/source/dmbok.pdf#7",
            "generation": 7,
            "sha256": SOURCE_SHA256,
        },
        "map": {
            "artifact_uri": "gs://corpus/maps/reviewed.json",
            "content_hash": map_content_hash,
            "file_sha256": map_sha256,
            "generation": "12",
            "map_id": "reviewed-map",
            "schema_version": "structure-map-v1",
        },
        "ingestion_gate": INGESTION_GATE,
    }
    approval_bytes = json.dumps(approval, sort_keys=True).encode()
    approval_sha256 = hashlib.sha256(approval_bytes).hexdigest()
    return map_bytes, approval_bytes, map_sha256, approval_sha256


def _source_manifest():
    return create_source_manifest(
        bucket="corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=SOURCE_SHA256,
    )


def _report():
    source = _source_manifest()
    return ExtractionReport(
        source_uri=source.object_uri,
        source_sha256=SOURCE_SHA256,
        parser="test parser",
        page_count=2,
        pages=(
            PageProvenance(1, "Chapter 1", "First reviewed passage."),
            PageProvenance(2, "Chapter 1", "Second reviewed passage."),
        ),
        failures=(),
    )


def _build_candidate(*, map_content_hash: str = "b" * 64):
    map_bytes, approval_bytes, map_sha256, approval_sha256 = _artifacts(
        map_content_hash=map_content_hash
    )
    return entrypoint.build_hierarchical_candidate(
        source_manifest=_source_manifest(),
        map_bytes=map_bytes,
        map_uri=MAP_URI,
        expected_map_sha256=map_sha256,
        approval_bytes=approval_bytes,
        approval_uri=APPROVAL_URI,
        expected_approval_sha256=approval_sha256,
        vector_collection="document_chunks_v2",
        extract=_report,
    )


def test_map_verification_failure_leaves_active_corpus_unchanged() -> None:
    versions = InMemoryCorpusVersionStore()
    legacy = CorpusVersion(
        version_id="legacy",
        source_uri="gs://corpus/legacy.pdf#1",
        embedding_model="gemini-embedding-001",
        status="staged",
        created_at=datetime.now(timezone.utc),
    )
    versions.stage(legacy)
    versions.validate(
        "legacy",
        [
            TextChunk(
                chunk_id="legacy:1:0",
                corpus_version_id="legacy",
                page=1,
                section="Old",
                chunk_ordinal=0,
                content_hash="c" * 64,
                text="legacy passage",
            )
        ],
    )
    versions.activate("legacy")
    active_before = versions.active()
    map_bytes, approval_bytes, _, approval_sha256 = _artifacts()

    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        entrypoint.build_and_stage_hierarchical_candidate(
            versions=versions,
            source_manifest=_source_manifest(),
            map_bytes=map_bytes,
            map_uri=MAP_URI,
            expected_map_sha256="d" * 64,
            approval_bytes=approval_bytes,
            approval_uri=APPROVAL_URI,
            expected_approval_sha256=approval_sha256,
            vector_collection="document_chunks_v2",
            extract=_report,
        )

    assert versions.active() == active_before
    assert versions.get("legacy") == active_before
    assert len(versions._versions) == 1


def test_repeated_derivation_resumes_same_staged_identity_without_activating() -> None:
    versions = InMemoryCorpusVersionStore()
    first = entrypoint.build_and_stage_hierarchical_candidate(versions=versions, **_candidate_args())
    retry = entrypoint.build_and_stage_hierarchical_candidate(versions=versions, **_candidate_args())

    assert first.version.version_id == retry.version.version_id
    assert first.manifest.manifest_sha256 == retry.manifest.manifest_sha256
    assert tuple(parent.parent_id for parent in first.parents) == tuple(
        parent.parent_id for parent in retry.parents
    )
    assert tuple(child.chunk_id for child in first.children) == tuple(
        child.chunk_id for child in retry.children
    )
    assert all(parent.corpus_version_id == first.version.version_id for parent in first.parents)
    assert all(child.corpus_version_id == first.version.version_id for child in first.children)
    assert versions.active() is None
    assert versions.get(first.version.version_id).status == "staged"  # type: ignore[union-attr]
    assert first.ingestion_gate == INGESTION_GATE


def _candidate_args():
    map_bytes, approval_bytes, map_sha256, approval_sha256 = _artifacts()
    return {
        "source_manifest": _source_manifest(),
        "map_bytes": map_bytes,
        "map_uri": MAP_URI,
        "expected_map_sha256": map_sha256,
        "approval_bytes": approval_bytes,
        "approval_uri": APPROVAL_URI,
        "expected_approval_sha256": approval_sha256,
        "vector_collection": "document_chunks_v2",
        "extract": _report,
    }


def test_changed_map_token_counter_or_splitter_creates_new_version_and_ids(monkeypatch) -> None:
    import app.corpus.manifest as manifest_module

    base = _build_candidate()
    changed_map = _build_candidate(map_content_hash="e" * 64)
    monkeypatch.setattr(manifest_module, "TOKENIZER_PACKAGE_VERSION", "0.13.0")
    changed_tokenizer = entrypoint.build_hierarchical_candidate(**_candidate_args())
    monkeypatch.setattr(manifest_module, "CHILD_SEGMENTATION_VERSION", "stable-child-v2")
    changed_splitter = entrypoint.build_hierarchical_candidate(**_candidate_args())

    for changed in (changed_map, changed_tokenizer, changed_splitter):
        assert changed.version.version_id != base.version.version_id
        assert changed.parents[0].parent_id != base.parents[0].parent_id
        assert changed.children[0].chunk_id != base.children[0].chunk_id


def test_map_approval_must_bind_exact_source_and_map_identity() -> None:
    map_bytes, approval_bytes, map_sha256, approval_sha256 = _artifacts()
    approval = json.loads(approval_bytes)
    approval["source"]["generation"] = 8
    altered_approval = json.dumps(approval, sort_keys=True).encode()

    with pytest.raises(RuntimeError, match="immutable PDF source"):
        entrypoint.build_hierarchical_candidate(
            source_manifest=_source_manifest(),
            map_bytes=map_bytes,
            map_uri=MAP_URI,
            expected_map_sha256=map_sha256,
            approval_bytes=altered_approval,
            approval_uri=APPROVAL_URI,
            expected_approval_sha256=hashlib.sha256(altered_approval).hexdigest(),
            vector_collection="document_chunks_v2",
            extract=_report,
        )


def test_oversized_indivisible_source_unit_remains_a_blocking_candidate_exception() -> None:
    versions = InMemoryCorpusVersionStore()
    report = _report()
    pages = (
        PageProvenance(1, "Chapter 1", "- " + " ".join(["item"] * 1_600)),
        PageProvenance(2, "Chapter 1", "Retained searchable passage."),
    )
    oversized_report = ExtractionReport(
        source_uri=report.source_uri,
        source_sha256=report.source_sha256,
        parser=report.parser,
        page_count=2,
        pages=pages,
        failures=(),
    )
    args = _candidate_args()
    args["extract"] = lambda: oversized_report
    candidate = entrypoint.build_and_stage_hierarchical_candidate(versions=versions, **args)

    assert candidate.exceptions
    assert candidate.exceptions[0].reason == "indivisible list item or table row exceeds parent maximum"
    assert candidate.version.status == "staged"
    assert versions.active() is None
