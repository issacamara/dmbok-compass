import copy
import hashlib
import json

import pytest

from app.corpus import create_source_manifest, default_derivation


PDF_HASH = "a" * 64


def test_manifest_binds_version_to_immutable_source_generation() -> None:
    manifest = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=PDF_HASH,
    )

    assert manifest.object_uri == "gs://dmbok-corpus/source/dmbok.pdf#7"
    assert manifest.corpus_version_id.startswith("v-")


def test_replacing_source_object_creates_a_new_corpus_version() -> None:
    original = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=PDF_HASH,
    )
    replacement = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=8,
        content_sha256="b" * 64,
    )

    assert replacement.corpus_version_id != original.corpus_version_id


def _set(material: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target: dict[str, object] = material
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment, index]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("implementation_revision",), "langchain-rag-v3"),
        (("loader", "type"), "other.PdfLoader"),
        (("loader", "loader_version"), "0.4.3"),
        (("loader", "parser"), "other-pdf-parser"),
        (("loader", "parser_version"), "6.20.0"),
        (("loader", "options", "mode"), "single"),
        (("normalization", "version"), "whitespace-v2"),
        (("provenance", "version"), "section-recovery-v2"),
        (("splitter", "type"), "other.Splitter"),
        (("splitter", "version"), "1.1.4"),
        (("splitter", "units"), "tokens"),
        (("splitter", "chunk_size"), 701),
        (("splitter", "chunk_overlap"), 69),
        (("splitter", "separator"), "\\n"),
        (("embedding", "model"), "gemini-embedding-002"),
        (("embedding", "dimensions"), 1024),
        (("embedding", "document_task"), "SEMANTIC_SIMILARITY"),
        (("embedding", "query_task"), "CLUSTERING"),
        (("embedding", "integration_version"), "4.4.2"),
        (("vector_schema", "collection"), "document_chunks_v3"),
        (("vector_schema", "version"), "3"),
        (("vector_schema", "content_field"), "contents"),
        (("vector_schema", "metadata_field"), "attributes"),
        (("vector_schema", "embedding_field"), "vector"),
    ],
)
def test_each_required_derivation_input_changes_the_version(
    path: tuple[str, ...], value: object
) -> None:
    original = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=PDF_HASH,
    )
    changed_derivation = copy.deepcopy(default_derivation())
    _set(changed_derivation, path, value)

    changed = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=PDF_HASH,
        derivation=changed_derivation,
    )

    assert changed.corpus_version_id != original.corpus_version_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bucket", "other-corpus"),
        ("object_name", "source/revised-dmbok.pdf"),
        ("generation", 8),
        ("content_sha256", "b" * 64),
    ],
)
def test_each_source_identity_input_changes_the_version(field: str, value: object) -> None:
    source: dict[str, object] = {
        "bucket": "dmbok-corpus",
        "object_name": "source/dmbok.pdf",
        "generation": 7,
        "content_sha256": PDF_HASH,
    }
    original = create_source_manifest(**source)  # type: ignore[arg-type]
    source[field] = value
    changed = create_source_manifest(**source)  # type: ignore[arg-type]

    assert changed.corpus_version_id != original.corpus_version_id


def test_identical_derivation_is_canonical_across_retries_and_dictionary_ordering() -> None:
    derivation = default_derivation()
    reordered = json.loads(json.dumps(derivation, sort_keys=True, indent=2), object_pairs_hook=dict)
    first = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=PDF_HASH,
        derivation=derivation,
    )
    retry = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=PDF_HASH,
        derivation=reordered,
    )

    assert retry.corpus_version_id == first.corpus_version_id
    assert retry.manifest_sha256 == first.manifest_sha256


def test_manifest_metadata_records_reproducible_material_and_hashes_without_secrets() -> None:
    manifest = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256=PDF_HASH,
    )
    metadata = manifest.corpus_metadata()
    material = metadata["corpus_manifest_material"]

    assert metadata["corpus_manifest"] == json.loads(material)
    assert metadata["corpus_manifest_sha256"] == hashlib.sha256(material.encode()).hexdigest()
    assert metadata["derivation_sha256"] == hashlib.sha256(manifest.derivation.encode()).hexdigest()
    assert "secret" not in material.lower()
    assert "token" not in material.lower()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation", 0),
        ("content_sha256", "not-a-digest"),
        ("content_type", "text/plain"),
        ("object_name", "source/dmbok.txt"),
    ],
)
def test_manifest_rejects_unapproved_source_metadata(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "bucket": "dmbok-corpus",
        "object_name": "source/dmbok.pdf",
        "generation": 7,
        "content_sha256": PDF_HASH,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        create_source_manifest(**kwargs)  # type: ignore[arg-type]


def test_manifest_rejects_missing_required_derivation_field_and_secret_metadata() -> None:
    incomplete = default_derivation()
    del incomplete["embedding"]

    with pytest.raises(ValueError, match="embedding"):
        create_source_manifest(
            bucket="dmbok-corpus",
            object_name="source/dmbok.pdf",
            generation=7,
            content_sha256=PDF_HASH,
            derivation=incomplete,
        )

    with pytest.raises(ValueError, match="secrets"):
        create_source_manifest(
            bucket="dmbok-corpus",
            object_name="source/dmbok.pdf",
            generation=7,
            content_sha256=PDF_HASH,
            derivation={"api_key": "must-not-be-persisted"},
        )
