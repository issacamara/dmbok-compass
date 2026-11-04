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
