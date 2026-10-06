import base64
from pathlib import Path

from app.ingestion import extract_pdf


FIXTURE = Path(__file__).parent / "fixtures/corpus/searchable-provenance-fixture.pdf.b64"


def _materialise_fixture(tmp_path: Path) -> Path:
    output = tmp_path / "source.pdf"
    output.write_bytes(base64.b64decode(FIXTURE.read_text()))
    return output


def test_extracts_one_based_pages_and_carries_section_provenance(tmp_path: Path) -> None:
    report = extract_pdf(_materialise_fixture(tmp_path), source_uri="gs://bucket/source.pdf#7")

    assert report.page_count == 2
    assert report.pages[0].page == 1
    assert report.pages[0].section == "1.1 Purpose"
    assert report.pages[1].page == 2
    assert report.pages[1].section == "1.2 Roles"
    assert report.failures == ()
    assert report.section_recovery_rate == 1.0


def test_report_is_reproducible_and_binds_source_identity(tmp_path: Path) -> None:
    pdf = _materialise_fixture(tmp_path)
    first = extract_pdf(pdf, source_uri="gs://bucket/source.pdf#7")
    second = extract_pdf(pdf, source_uri="gs://bucket/source.pdf#7")

    assert first.to_json() == second.to_json()
    assert first.source_uri == "gs://bucket/source.pdf#7"
    assert len(first.source_sha256) == 64

