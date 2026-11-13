from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import sys

import pytest
from app.performance import AUTHORITY_REF, load_authoritative_release_cost_record

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.run_performance_probe import _jwt_subject, _redacted_base_url, _safe_evidence, parse_args, probe_user, run_probe


class Response:
    status_code = 200
    headers = {"content-type": "application/json"}

    def json(self) -> dict[str, object]:
        return {"outcome": "answer", "usage": {}, "trace": {}}


class Client:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def post(self, path: str, *, headers: dict[str, str], json: dict[str, object]) -> Response:
        self.requests.append({"path": path, "headers": headers, "json": json})
        return Response()


def test_probe_passes_expected_outcome_and_stores_it_on_each_sample() -> None:
    client = Client()
    samples = asyncio.run(probe_user(client, 0, "opaque-token", "private question", 1, "refusal", "subject-1"))

    assert client.requests[0]["json"] == {"question": "private question"}
    assert samples[0].expected_outcome == "refusal"
    assert samples[0].outcome == "answer"
    assert samples[0].user_subject == "subject-1"


def _token(subject: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"sub": subject}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_probe_rejects_distinct_tokens_with_same_subject_before_any_request() -> None:
    class NeverClient:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("API client must not be created for duplicate subjects")

    args = type("Args", (), {"token": [_token("same-user"), _token("same-user"), _token("third-user")]})()
    import scripts.run_performance_probe as probe
    original = probe.httpx.AsyncClient
    probe.httpx.AsyncClient = NeverClient
    try:
        with pytest.raises(ValueError, match="three distinct users"):
            asyncio.run(run_probe(args))
    finally:
        probe.httpx.AsyncClient = original


def test_jwt_subject_is_read_without_requiring_token_signature_validation() -> None:
    assert _jwt_subject(_token("private-subject")) == "private-subject"


def test_base_url_assumption_redacts_host_path_and_query() -> None:
    rendered = _redacted_base_url("https://secret-project.run.app/private/path?token=secret")

    assert rendered == "https://[redacted]"
    assert "secret-project" not in rendered
    assert "private" not in rendered
    assert "token" not in rendered


def test_probe_has_no_zero_default_rate_or_free_form_gcp_cost_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "run_performance_probe.py", "--base-url", "https://preview.example.com",
        "--token", "one", "--token", "two", "--token", "three",
    ])

    args = parse_args()

    assert not hasattr(args, "cost_evidence")
    assert not hasattr(args, "provider_cap_approval_evidence")
    assert not hasattr(args, "primary_input_eur_per_million")
    assert not hasattr(args, "gcp_cost_forecast_eur")


def test_cost_authority_requires_a_committed_candidate_bound_repo_record(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate = {
        "artifact_id": "image-v1", "installed_lock_sha256": "a" * 64,
        "corpus_version_id": "corpus-v1", "configuration_version_id": "config-v1",
        "scorer_version": "scorer-v1",
    }
    record = {
        "schema_version": "release-cost-approval-v1", "candidate_identity": candidate,
        "sponsor_approval": {
            "approval_reference": "https://github.com/issacamara/dmbok-compass/issues/348",
            "approved_by": "sponsor-1", "approved_at": "2026-11-13T12:00:00Z",
            "provider": "openrouter", "monthly_cap_eur": 50,
        },
        "provider_rates": {
            "source_reference": "https://openrouter.ai/api/v1/models",
            "source_timestamp": "2026-11-13T12:00:00Z", "currency": "EUR",
            "primary_input_eur_per_million": 1, "primary_output_eur_per_million": 2,
            "fallback_input_eur_per_million": 3, "fallback_output_eur_per_million": 4,
        },
        "gcp_forecast": {
            "source_reference": "https://cloud.google.com/billing/docs",
            "source_timestamp": "2026-11-13T12:00:00Z", "monthly_cost_eur": 1.0,
        },
    }
    _init_authority_repo(repo)
    record_path = repo / "documents/decisions/release-cost-approval.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    # The feature branch's own HEAD is not authoritative without the trusted ref.
    assert load_authoritative_release_cost_record(repo, candidate) is None
    _git(repo, "add", "documents/decisions/release-cost-approval.json")
    _git(repo, "commit", "-m", "add synthetic authority record")
    trusted_commit = _git_output(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", AUTHORITY_REF, trusted_commit)
    _git(repo, "checkout", "-b", "feature/test")
    record["sponsor_approval"]["monthly_cap_eur"] = 999
    record_path.write_text(json.dumps(record), encoding="utf-8")
    _git(repo, "commit", "-am", "add feature-branch cap claim")

    verified = load_authoritative_release_cost_record(repo, candidate)
    assert verified is not None
    assert verified.is_verified_for(candidate)
    assert verified["authority_path"] == "documents/decisions/release-cost-approval.json"
    assert verified["sponsor_approval"]["monthly_cap_eur"] == 50
    assert verified.source_commit == trusted_commit
    assert verified["authority_ref"] == AUTHORITY_REF
    assert load_authoritative_release_cost_record(repo, {**candidate, "artifact_id": "other"}) is None


def test_cost_authority_rejects_invented_sources_even_when_committed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_authority_repo(repo)
    candidate = {
        "artifact_id": "image-v1", "installed_lock_sha256": "a" * 64,
        "corpus_version_id": "corpus-v1", "configuration_version_id": "config-v1",
        "scorer_version": "scorer-v1",
    }
    record = _synthetic_cost_record(candidate)
    record["provider_rates"]["source_reference"] = "https://example.org/fake-prices"
    record_path = repo / "documents/decisions/release-cost-approval.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    _git(repo, "add", "documents/decisions/release-cost-approval.json")
    _git(repo, "commit", "-m", "add invalid authority record")
    _git(repo, "update-ref", AUTHORITY_REF, "HEAD")

    assert load_authoritative_release_cost_record(repo, candidate) is None


def test_cost_authority_rejects_trusted_ref_that_is_not_an_ancestor(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_authority_repo(repo)
    candidate = {
        "artifact_id": "image-v1", "installed_lock_sha256": "a" * 64,
        "corpus_version_id": "corpus-v1", "configuration_version_id": "config-v1",
        "scorer_version": "scorer-v1",
    }
    _git(repo, "checkout", "-b", "feature/diverged")
    (repo / "feature.txt").write_text("feature commit\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "diverge feature")
    _git(repo, "checkout", "development")
    record_path = repo / "documents/decisions/release-cost-approval.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(_synthetic_cost_record(candidate)), encoding="utf-8")
    _git(repo, "add", "documents/decisions/release-cost-approval.json")
    _git(repo, "commit", "-m", "add trusted authority")
    _git(repo, "update-ref", AUTHORITY_REF, "HEAD")
    _git(repo, "checkout", "feature/diverged")

    assert load_authoritative_release_cost_record(repo, candidate) is None


def test_imported_quality_json_cannot_claim_unverified_split_binding(tmp_path: Path) -> None:
    evidence_path = tmp_path / "quality.json"
    thresholds = {
        "retrieval_success": 90, "grounded_claims": 95,
        "citation_correctness": 95, "answer_quality": 85,
        "refusal_correctness": 95, "response_time": 95,
    }
    evidence = {
        "candidate": {
            "artifact_id": "image-v1", "installed_lock_sha256": "a" * 64,
            "corpus_version_id": "corpus-v1", "configuration_version_id": "config-v1",
            "scorer_version": "scorer-v1",
        },
        "held_out_digest": "b" * 64, "tuning_digest": "c" * 64,
        "split_binding_status": "verified", "model_overlap": None,
        "metrics": {
            name: {
                "numerator": threshold, "denominator": 100, "percentage": float(threshold),
                "threshold": threshold, "passed": True,
            }
            for name, threshold in thresholds.items()
        },
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    imported = _safe_evidence(evidence_path, "quality")

    assert imported["split_binding_status"] == "unverified"
    assert imported["quality_metrics_passed"] is True


def test_stage_evidence_requires_costs_with_usage(tmp_path: Path) -> None:
    evidence_path = tmp_path / "stages.json"
    evidence = {
        stage: {
            "status": "complete", "used_units": 10, "reserved_units": 12,
            "cost_eur": 0.2, "reserved_cost_eur": 0.3,
        }
        for stage in ("parser", "generator", "judge")
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    assert _safe_evidence(evidence_path, "stages") == evidence
    del evidence["judge"]["cost_eur"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid shape"):
        _safe_evidence(evidence_path, "stages")

    evidence["judge"] = {
        "status": "unknown", "used_units": None, "reserved_units": None,
        "cost_eur": None, "reserved_cost_eur": None,
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert _safe_evidence(evidence_path, "stages")["judge"]["status"] == "unknown"


def _init_authority_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "remote", "add", "origin", "https://github.com/issacamara/dmbok-compass.git")
    _git(repo, "config", "user.email", "codex-test@example.invalid")
    _git(repo, "config", "user.name", "Codex test")
    (repo / "README.md").write_text("fixture repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initialize")
    _git(repo, "branch", "-M", "development")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _synthetic_cost_record(candidate: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": "release-cost-approval-v1", "candidate_identity": candidate,
        "sponsor_approval": {
            "approval_reference": "https://github.com/issacamara/dmbok-compass/issues/348",
            "approved_by": "sponsor-1", "approved_at": "2026-11-13T12:00:00Z",
            "provider": "openrouter", "monthly_cap_eur": 50,
        },
        "provider_rates": {
            "source_reference": "https://openrouter.ai/api/v1/models",
            "source_timestamp": "2026-11-13T12:00:00Z", "currency": "EUR",
            "primary_input_eur_per_million": 1, "primary_output_eur_per_million": 2,
            "fallback_input_eur_per_million": 3, "fallback_output_eur_per_million": 4,
        },
        "gcp_forecast": {
            "source_reference": "https://cloud.google.com/billing/docs",
            "source_timestamp": "2026-11-13T12:00:00Z", "monthly_cost_eur": 1.0,
        },
    }
