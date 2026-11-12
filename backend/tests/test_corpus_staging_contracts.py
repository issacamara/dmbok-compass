from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import CorpusStagingAttemptStatus, StructureTemplate
from app.contracts.api import ApiError


FIXTURES = Path(__file__).parents[2] / "documents/contracts"


def test_structure_template_and_attempt_fixtures_validate() -> None:
    template = StructureTemplate.model_validate_json(
        (Path(__file__).parents[1] / "app/templates/dmbok-structure-template-v1.json").read_text()
    )
    assert template.entries[0].section_path == ("Chapter 1: Data Management", "1.1 Introduction")

    fixture = json.loads((FIXTURES / "corpus-staging-contract-fixture.json").read_text())
    for key in ("valid", "invalid", "terminal"):
        attempt = CorpusStagingAttemptStatus.model_validate(fixture["attempts"][key])
        assert attempt.state in {"awaiting_review", "failed", "staged"}
    assert ApiError.model_validate(fixture["attempts"]["stale"]).status == 409


@pytest.mark.parametrize("field", ["object_uri", "generation", "sha256", "service_identity"])
def test_browser_staging_contract_rejects_server_owned_identity(field: str) -> None:
    fixture = json.loads((FIXTURES / "corpus-staging-contract-fixture.json").read_text())
    attempt = fixture["attempts"]["valid"] | {field: "must-not-cross-browser-contract"}
    with pytest.raises(ValidationError):
        CorpusStagingAttemptStatus.model_validate(attempt)

    template = json.loads(
        (Path(__file__).parents[1] / "app/templates/dmbok-structure-template-v1.json").read_text()
    ) | {field: "must-not-be-entered-by-admin"}
    with pytest.raises(ValidationError):
        StructureTemplate.model_validate(template)


def test_staging_contract_rejects_invalid_ranges_and_incomplete_terminal_state() -> None:
    with pytest.raises(ValidationError):
        StructureTemplate.model_validate({
            "schema_version": "structure-template-v1",
            "document_title": "DMBOK",
            "instructions": ["Example"],
            "entries": [{"section_path": ["Chapter 1"], "page_start": 3, "page_end": 2}],
        })

    with pytest.raises(ValidationError):
        CorpusStagingAttemptStatus(
            attempt_id="attempt-staged", revision=1, state="staged", chunk_count=2
        )


def test_staging_status_exposes_safe_parse_and_candidate_generation_progress() -> None:
    status = CorpusStagingAttemptStatus(
        attempt_id="attempt-review", revision=1, state="awaiting_review",
        preview=[], parsing_page_count=12, parsing_block_count=38,
        parsing_usage_kind="credits", parsing_usage_units=15,
        candidate_generation_state="running", candidate_generation_count=40,
        candidate_generation_target=200, candidate_generation_rejected_count=3,
        candidate_generation_usage_tokens=9000,
    )

    payload = status.model_dump(mode="json", exclude_none=True)
    assert payload["parsing_page_count"] == 12
    assert payload["parsing_usage_kind"] == "credits"
    assert payload["candidate_generation_count"] == 40
    assert payload["candidate_generation_target"] == 200
    assert payload["candidate_generation_usage_tokens"] == 9000

    with pytest.raises(ValidationError):
        CorpusStagingAttemptStatus(
            attempt_id="attempt-review", revision=1, state="awaiting_review",
            preview=[], parsing_usage_units=0,
        )
