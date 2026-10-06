import pytest

from app.corpus import SourcePage, chunk_pages


def test_chunking_never_crosses_pages_and_preserves_provenance() -> None:
    pages = [
        SourcePage(page=2, section="Data governance", text="one two three four five"),
        SourcePage(page=3, section="Metadata", text="six seven"),
    ]

    chunks = chunk_pages(pages, corpus_version_id="v-abc", target_tokens=3, overlap_tokens=1)

    assert [(chunk.page, chunk.section, chunk.chunk_ordinal) for chunk in chunks] == [
        (2, "Data governance", 0),
        (2, "Data governance", 1),
        (3, "Metadata", 0),
    ]
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five", "six seven"]
    assert all(chunk.corpus_version_id == "v-abc" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("v-abc:") for chunk in chunks)


def test_chunking_normalizes_whitespace_and_is_repeatable() -> None:
    page = SourcePage(page=1, section="  Overview  ", text=" alpha\n\tbeta  gamma ")

    first = chunk_pages([page], corpus_version_id="v-1", target_tokens=2, overlap_tokens=0)
    second = chunk_pages([page], corpus_version_id="v-1", target_tokens=2, overlap_tokens=0)

    assert first == second
    assert [chunk.text for chunk in first] == ["alpha beta", "gamma"]
    assert first[0].section == "Overview"
    assert len(first[0].content_hash) == 64


def test_empty_pages_are_ignored() -> None:
    chunks = chunk_pages(
        [SourcePage(page=1, section="Empty", text="\n  "), SourcePage(page=2, section="Body", text="kept")],
        corpus_version_id="v-1",
    )

    assert [chunk.page for chunk in chunks] == [2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_tokens": 0}, "target_tokens"),
        ({"target_tokens": 4, "overlap_tokens": 4}, "overlap_tokens"),
        ({"target_tokens": 4, "overlap_tokens": -1}, "overlap_tokens"),
        ({"corpus_version_id": " v-1"}, "corpus_version_id"),
    ],
)
def test_chunking_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    defaults: dict[str, object] = {"corpus_version_id": "v-1"}
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        chunk_pages([SourcePage(page=1, section="Body", text="text")], **defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "page",
    [
        SourcePage(page=0, section="Body", text="text"),
        SourcePage(page=1, section=" ", text="text"),
    ],
)
def test_chunking_rejects_invalid_page_provenance(page: SourcePage) -> None:
    with pytest.raises(ValueError):
        chunk_pages([page], corpus_version_id="v-1")
