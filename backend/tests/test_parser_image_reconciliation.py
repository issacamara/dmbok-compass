import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[2] / "scripts/reconcile_parser_image.py"
spec = importlib.util.spec_from_file_location("reconcile_parser_image", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run_reconciliation(monkeypatch, images):
    monkeypatch.setenv("PROJECT_ID", "development-project")
    images = iter(images)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ("run", "services", "describe"):
            return SimpleNamespace(stdout="api-revision\n")
        if args[1:4] == ("run", "revisions", "describe"):
            return SimpleNamespace(stdout=next(images) + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.main()
    return [call for call in calls if call[1:4] == ("run", "jobs", "update")]


def test_provision_after_delivery_reconciles_current_digest(monkeypatch):
    image = "registry.example/api@sha256:" + "a" * 64
    updates = run_reconciliation(monkeypatch, [image, image])
    assert len(updates) == 1
    assert f"--image={image}" in updates[0]
    assert "--project=development-project" in updates[0]


def test_changed_api_image_is_reconciled_again(monkeypatch):
    old = "registry.example/api@sha256:" + "a" * 64
    new = "registry.example/api@sha256:" + "b" * 64
    updates = run_reconciliation(monkeypatch, [old, new, new, new])
    assert [call[-1] for call in updates] == [f"--image={old}", f"--image={new}"]


def test_first_provisioning_keeps_placeholder_until_delivery(monkeypatch):
    assert run_reconciliation(monkeypatch, ["us-docker.pkg.dev/cloudrun/container/hello"]) == []


def test_unpinned_api_image_fails_actionably(monkeypatch):
    with pytest.raises(RuntimeError, match="not digest-pinned"):
        run_reconciliation(monkeypatch, ["registry.example/api:latest"])


def test_continuously_changing_api_fails_instead_of_silently_diverging(monkeypatch):
    images = [f"registry.example/api@sha256:{digit * 64}" for digit in "abcdef"]
    with pytest.raises(RuntimeError, match="changed during reconciliation"):
        run_reconciliation(monkeypatch, images)


def test_reconciliation_uses_protected_environment_project():
    workflow = (PATH.parents[1] / ".github/workflows/terraform.yml").read_text()
    development, production = workflow.split("  apply-production:", 1)
    assert "python scripts/reconcile_parser_image.py" in development
    assert "python scripts/reconcile_parser_image.py" in production
    assert "environment: production" in production
    assert "PROJECT_ID: ${{ vars.GCP_PROJECT_PRODUCTION }}" in production
