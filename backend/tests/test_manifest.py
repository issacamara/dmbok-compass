import pytest

from app.corpus import create_source_manifest


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
