from fastapi.testclient import TestClient

from app.contracts.api import AggregateMetric, UserProfile
from app.identity import IdentityError, get_admin_user
from app.main import aggregate_metric_repository, app
from app.metrics import InMemoryAggregateMetricRepository


def metric(name: str, numerator: int, denominator: int, percentage: float) -> AggregateMetric:
    return AggregateMetric(
        metric_name=name,
        numerator=numerator,
        denominator=denominator,
        percentage=percentage,
    )


def test_in_memory_metrics_are_sorted_and_content_free() -> None:
    repository = InMemoryAggregateMetricRepository(
        [
            metric("provider_failures", 2, 20, 10),
            metric("request_count", 20, 1, 100),
        ]
    )

    assert [item.metric_name for item in repository.list_metrics()] == [
        "provider_failures",
        "request_count",
    ]
    repository.replace(metric("fallback_rate", 3, 20, 15))
    assert {item.metric_name for item in repository.list_metrics()} == {
        "fallback_rate",
        "provider_failures",
        "request_count",
    }

    persisted = repository.replace(metric("estimated_cost_cents", 42, 100, 42))
    assert set(persisted.model_dump()) == {"metric_name", "numerator", "denominator", "percentage"}


def test_admin_metrics_endpoint_returns_only_aggregate_contract() -> None:
    repository = InMemoryAggregateMetricRepository(
        [metric("request_count", 20, 1, 100), metric("quota_consumption", 7, 100, 7)]
    )
    admin = UserProfile(
        user_id="admin",
        email="admin@example.com",
        username="Admin",
        approval_state="approved",
        role="admin",
        email_verified=True,
    )
    app.dependency_overrides[get_admin_user] = lambda: admin
    app.dependency_overrides[aggregate_metric_repository] = lambda: repository
    try:
        response = TestClient(app).get("/api/admin/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == [
        {"metric_name": "quota_consumption", "numerator": 7, "denominator": 100, "percentage": 7.0},
        {"metric_name": "request_count", "numerator": 20, "denominator": 1, "percentage": 100.0},
    ]
    assert all(
        field not in response.text
        for field in ("question", "answer", "passage", "trace", "excerpt")
    )


def test_admin_metrics_endpoint_requires_admin_dependency() -> None:
    def deny() -> UserProfile:
        raise IdentityError("Administrator access is required.", 403)

    app.dependency_overrides[get_admin_user] = deny
    app.dependency_overrides[aggregate_metric_repository] = lambda: InMemoryAggregateMetricRepository()
    try:
        response = TestClient(app).get("/api/admin/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
