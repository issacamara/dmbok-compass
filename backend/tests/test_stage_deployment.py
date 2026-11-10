from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/deploy_cloud_run.sh"


def deploy(tmp_path: Path, **settings: str) -> subprocess.CompletedProcess[str]:
    fake = tmp_path / "gcloud"
    fake.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [ "$1 $2 $3" = "run jobs list" ]; then\n'
        '  [ "${FAKE_LIST_FAIL:-0}" = 0 ] || exit 1\n'
        '  if [ "${FAKE_DELAYED_JOB:-0}" = 1 ]; then\n'
        '    if [ -e "$CALL_LOG.listed" ]; then printf "%s\\n" dmbok-compass-parser; fi\n'
        '    touch "$CALL_LOG.listed"; exit 0\n'
        '  fi\n'
        '  [ "${FAKE_PARSER_JOB:-0}" = 0 ] || printf "%s\\n" dmbok-compass-parser\n'
        'fi\n'
    )
    fake.chmod(0o755)
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALL_LOG": str(tmp_path / "calls"),
        "PROJECT_ID": "project",
        "API_IMAGE": "image",
        "SOURCE_SHA": "a" * 40,
        "DEPLOY_SHA": "b" * 40,
        "CORPUS_VERSION_ID": "corpus",
        "CORPUS_BUCKET_NAME": "bucket",
        "CONFIGURATION_VERSION_ID": "config",
        "RUNTIME_SECRET_VERSION": "1",
        **settings,
    }
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False)


def test_parser_secret_is_only_delivered_to_isolated_parser(tmp_path: Path) -> None:
    result = deploy(
        tmp_path,
        PARSER_SECRET_VERSION="3",
        PARSER_ROUTE="llamacloud",
        PARSER_MODE="parse",
        PARSER_ALLOWANCE_POLICY="observed",
        PARSER_ALLOWANCE_MAX_UNITS="50",
        FAKE_PARSER_JOB="1",
    )
    assert result.returncode == 0, result.stderr
    preflight, api, ingestion, evaluation, parser = (tmp_path / "calls").read_text().splitlines()
    assert "run jobs list" in preflight
    assert all("--image=image" in call for call in (api, ingestion, evaluation, parser))
    assert "LLAMACLOUD_API_KEY=llamacloud-parser-api-key:3" in parser
    assert "--service-account=" not in parser
    assert "LLAMACLOUD_API_KEY=" not in api + ingestion + evaluation
    assert "PARSER_ROUTE=" not in api + ingestion + evaluation


def test_api_cannot_act_as_or_invoke_parser_identity() -> None:
    terraform = SCRIPT.parents[1] / "infra/terraform"
    iam = (terraform / "iam.tf").read_text()
    delivery = (terraform / "delivery.tf").read_text()
    assert 'service_account_id = google_service_account.workload["parser"].name' not in iam
    assert 'name     = google_cloud_run_v2_job.parser.name' not in delivery
    assert 'args    = ["-c", "raise SystemExit(\'Hosted parsing is not configured\')"]' in delivery


def test_missing_or_unpinned_stage_settings_fail_before_deployment(tmp_path: Path) -> None:
    result = deploy(tmp_path, PARSER_SECRET_VERSION="latest")
    assert result.returncode != 0
    assert not (tmp_path / "calls").exists()
    assert deploy(tmp_path, GENERATOR_ROUTE="openrouter").returncode != 0
    assert not (tmp_path / "calls").exists()


def test_unconfigured_parser_is_removed_and_serving_still_deploys(tmp_path: Path) -> None:
    assert deploy(tmp_path).returncode == 0
    preflight, api, ingestion, evaluation, second_check = (tmp_path / "calls").read_text().splitlines()
    assert "run jobs list" in preflight
    assert "run jobs list" in second_check
    assert all("--remove-secrets=LLAMACLOUD_API_KEY" in call for call in (api, ingestion, evaluation))
    assert "LLAMACLOUD_API_KEY=llamacloud-parser-api-key:" not in ingestion


def test_generator_and_judge_settings_are_only_sent_to_evaluation(tmp_path: Path) -> None:
    settings = {
        f"{stage}_{name}": value
        for stage in ("GENERATOR", "JUDGE")
        for name, value in {
            "ROUTE": "openrouter",
            "MODEL": "provider/model",
            "CONFIGURATION_VERSION": "v1",
            "ALLOWANCE_POLICY": "observed",
            "ALLOWANCE_MAX_UNITS": "100",
        }.items()
    }
    result = deploy(tmp_path, **settings)
    assert result.returncode == 0, result.stderr
    _, api, ingestion, evaluation, _ = (tmp_path / "calls").read_text().splitlines()
    assert "GENERATOR_MODEL=provider/model" in evaluation
    assert "JUDGE_MODEL=provider/model" in evaluation
    assert "GENERATOR_MODEL=" not in api + ingestion
    assert "JUDGE_MODEL=" not in api + ingestion


def test_disabled_parser_clears_credentials_and_updates_image(tmp_path: Path) -> None:
    configured = dict(FAKE_PARSER_JOB="1", API_IMAGE="old-image", PARSER_SECRET_VERSION="1",
                      PARSER_ROUTE="llamacloud", PARSER_MODE="parse", PARSER_ALLOWANCE_POLICY="observed",
                      PARSER_ALLOWANCE_MAX_UNITS="50")
    assert deploy(tmp_path, **configured).returncode == 0
    assert deploy(tmp_path, FAKE_PARSER_JOB="1", API_IMAGE="new-image").returncode == 0
    calls = (tmp_path / "calls").read_text().splitlines()
    assert "--update-secrets=LLAMACLOUD_API_KEY=llamacloud-parser-api-key:1" in calls[4]
    preflight, api, ingestion, evaluation, parser = calls[5:]
    assert "run jobs list" in preflight
    assert "--image=new-image" in parser
    assert "--service-account=" not in parser
    assert "--remove-secrets=LLAMACLOUD_API_KEY" in parser
    assert "--remove-env-vars=PARSER_SECRET_VERSION,PARSER_ROUTE,PARSER_MODE,PARSER_ALLOWANCE_POLICY,PARSER_ALLOWANCE_MAX_UNITS" in parser
    assert "--update-secrets=" not in parser


def test_missing_parser_and_list_failure_fail_before_deploying_when_configured(tmp_path: Path) -> None:
    settings = dict(PARSER_SECRET_VERSION="1", PARSER_ROUTE="llamacloud", PARSER_MODE="parse",
                    PARSER_ALLOWANCE_POLICY="observed", PARSER_ALLOWANCE_MAX_UNITS="50")
    assert deploy(tmp_path, **settings).returncode != 0
    assert len((tmp_path / "calls").read_text().splitlines()) == 1
    (tmp_path / "calls").unlink()
    assert deploy(tmp_path, FAKE_LIST_FAIL="1").returncode != 0
    assert len((tmp_path / "calls").read_text().splitlines()) == 1


def test_job_created_during_delivery_receives_exact_image(tmp_path: Path) -> None:
    image = "registry.example/app@sha256:" + "a" * 64
    result = deploy(tmp_path, FAKE_DELAYED_JOB="1", API_IMAGE=image)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert "run jobs list" in calls[0] and "run jobs list" in calls[4]
    assert "run jobs deploy dmbok-compass-parser" in calls[5]
    assert f"--image={image}" in calls[5]
