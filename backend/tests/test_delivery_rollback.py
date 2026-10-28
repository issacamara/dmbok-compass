"""Delivery rollback preflight tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_rollback_module():
    spec = importlib.util.spec_from_file_location(
        "delivery_rollback", ROOT / "scripts" / "delivery" / "rollback.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(schema: str = "v2", *, binding_schema: str | None = None) -> dict[str, object]:
    binding_schema = binding_schema or schema
    collection, index = {
        "v1": ("document_chunks", "document_chunks_vector"),
        "v2": ("document_chunks_v2", "document_chunks_v2_vector"),
    }[binding_schema]
    digest = "sha256:" + "a" * 64
    return {
        "commit": "b" * 40,
        "api_image": f"registry/api@{digest}",
        "admin_image": f"registry/api@{digest}",
        "api_digest": digest,
        "frontend_digest": "c" * 64,
        "api_schema_version": schema,
        "serving_binding": {
            "corpus_version_id": "corpus-2026-10-15",
            "collection": collection,
            "schema_version": binding_schema,
            "index": index,
            "configuration_digest": "d" * 64,
        },
        "preview_revision": "api-00042-abc",
        "release_revision": "api-00042-abc",
    }


def test_preflight_restores_complete_previous_then_forward_binding() -> None:
    rollback = load_rollback_module()
    candidate = manifest("v2")
    previous = manifest("v1")

    plan = rollback.build_plan(
        candidate, previous, protected_gate_approved=True, owner_approved=True
    )

    assert plan["rollback"]["serving_binding"]["schema_version"] == "v1"
    assert plan["forward"]["serving_binding"]["schema_version"] == "v2"
    assert plan["rollback"]["api_image"] == plan["rollback"]["admin_image"]


def test_preflight_rejects_v1_reader_bound_to_v2_data() -> None:
    rollback = load_rollback_module()
    invalid_previous = manifest("v1", binding_schema="v2")

    with pytest.raises(ValueError, match="API reader schema"):
        rollback.validate_manifest(invalid_previous)


@pytest.mark.parametrize("gate, owner", [(False, True), (True, False)])
def test_preflight_requires_existing_gate_and_owner(gate: bool, owner: bool) -> None:
    rollback = load_rollback_module()

    with pytest.raises(rollback.RollbackPreflightError):
        rollback.build_plan(manifest(), manifest("v1"), protected_gate_approved=gate, owner_approved=owner)


def test_both_deployments_use_one_image_for_api_and_jobs() -> None:
    development = (ROOT / ".github/workflows/cd-development.yml").read_text(encoding="utf-8")
    production = (ROOT / ".github/workflows/cd-production.yml").read_text(encoding="utf-8")
    deploy = (ROOT / "scripts/deploy_cloud_run.sh").read_text(encoding="utf-8")

    assert "bash scripts/deploy_cloud_run.sh" in development
    assert "bash scripts/deploy_cloud_run.sh" in production
    assert 'test "$copied_digest" = "$DIGEST"' in production
    assert deploy.count('--image="$API_IMAGE"') == 3
    assert "RAG_CORPUS_VERSION_ID=${CORPUS_VERSION_ID}" in deploy


def test_deployments_resolve_one_ready_corpus_before_mutation() -> None:
    development = (ROOT / ".github/workflows/cd-development.yml").read_text(encoding="utf-8")
    production = (ROOT / ".github/workflows/cd-production.yml").read_text(encoding="utf-8")
    terraform = (ROOT / ".github/workflows/terraform.yml").read_text(encoding="utf-8")
    manual_corpus_variable = "vars." + "CORPUS_VERSION_ID"

    for workflow in (development, production, terraform):
        assert "resolve_active_corpus.py" in workflow
        assert manual_corpus_variable not in workflow
    for workflow in (development, production):
        assert workflow.index("Resolve active ready corpus") < workflow.index("Deploy exact API image")
        assert "steps.resolve-corpus.outputs.corpus_version_id" in workflow
    assert terraform.count("TF_VAR_corpus_version_id=${{ steps.resolve-corpus.outputs.corpus_version_id }}") == 4


def test_terraform_couples_admin_images_and_retains_both_indexes() -> None:
    delivery = (ROOT / "infra" / "terraform" / "delivery.tf").read_text(encoding="utf-8")
    indexes = (ROOT / "infra" / "terraform" / "firestore_vector.tf").read_text(encoding="utf-8")

    assert delivery.count("var.api_image") == 3  # API, ingestion, and evaluation
    assert delivery.count('name  = "RAG_CORPUS_VERSION_ID"') == 3
    assert indexes.count("prevent_destroy = true") == 2


def test_terraform_configures_every_runtime_setting_required_by_evaluations() -> None:
    delivery = (ROOT / "infra" / "terraform" / "delivery.tf").read_text(encoding="utf-8")

    for name in (
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "RAG_VECTOR_COLLECTION",
        "RAG_VECTOR_SCHEMA_VERSION",
        "RAG_EMBEDDING_MODEL",
        "RAG_EMBEDDING_DIMENSIONS",
        "RAG_RETRIEVAL_STRATEGY",
        "RAG_CORPUS_VERSION_ID",
        "ANSWER_PROVIDER",
        "ANSWER_REQUEST_TIMEOUT_MS",
        "ANSWER_GENERATION_TIMEOUT_MS",
        "ANSWER_OUTPUT_TOKEN_CAP",
        "ANSWER_MAX_MODEL_ATTEMPTS",
        "OPENROUTER_JSON_SCHEMA_SUPPORTED",
        "OPENROUTER_PRIMARY_MODEL",
        "OPENROUTER_FALLBACK_MODEL",
        "RUNTIME_SECRET_VERSION",
    ):
        assert delivery.count(f'name  = "{name}"') >= 2


def test_terraform_allows_runtime_to_run_evaluation_job_with_overrides() -> None:
    delivery = (ROOT / "infra" / "terraform" / "delivery.tf").read_text(encoding="utf-8")

    assert 'resource "google_cloud_run_v2_job_iam_member" "runtime_evaluation_executor_with_overrides"' in delivery
    assert 'role     = "roles/run.jobsExecutorWithOverrides"' in delivery
    assert 'role     = "roles/run.invoker"' in delivery
    assert 'google_service_account.workload["runtime"].email' in delivery
