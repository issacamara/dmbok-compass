from fastapi.testclient import TestClient

from app.contracts import UserProfile
from app.contracts import CorpusStagingAttemptStatus
from app.api.corpus_staging import corpus_staging_approval_service
from app.identity import get_admin_user, get_current_profile
from app.main import app


def profile(role: str) -> UserProfile:
    return UserProfile(
        user_id="test-user", email="test@example.com", username="test-user",
        approval_state="approved", role=role, email_verified=True,
    )


def test_structure_template_download_is_admin_only_and_versioned() -> None:
    app.dependency_overrides.clear()
    try:
        client = TestClient(app)
        route = "/api/admin/corpus-staging/template"
        assert client.get(route).status_code == 401

        app.dependency_overrides[get_current_profile] = lambda: profile("user")
        assert client.get(route).status_code == 403

        app.dependency_overrides[get_admin_user] = lambda: profile("admin")
        response = client.get(route)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert "dmbok-structure-template-v1.json" in response.headers["content-disposition"]
        assert response.json()["schema_version"] == "structure-template-v1"
        assert response.json()["entries"][0]["page_start"] == 1
    finally:
        app.dependency_overrides.clear()


def test_approval_endpoint_is_admin_only_and_requires_revision() -> None:
    class ApprovalService:
        def __init__(self) -> None:
            self.calls = []

        def approve(self, *, attempt_id: str, expected_revision: int, actor_id: str) -> CorpusStagingAttemptStatus:
            self.calls.append((attempt_id, expected_revision, actor_id))
            return CorpusStagingAttemptStatus(attempt_id=attempt_id, revision=expected_revision + 1, state="queued", completed_stage="approval")

    app.dependency_overrides.clear()
    service = ApprovalService()
    app.dependency_overrides[corpus_staging_approval_service] = lambda: service
    try:
        client = TestClient(app)
        route = "/api/admin/corpus-staging/attempts/attempt-1/approve"
        assert client.post(route, params={"expected_revision": 4}).status_code == 401
        app.dependency_overrides[get_current_profile] = lambda: profile("user")
        assert client.post(route, params={"expected_revision": 4}).status_code == 403
        app.dependency_overrides[get_admin_user] = lambda: profile("admin")
        assert client.post(route).status_code == 422
        response = client.post(route, params={"expected_revision": 4})
        assert response.status_code == 200
        assert response.json()["state"] == "queued"
        assert service.calls == [("attempt-1", 4, "test-user")]
    finally:
        app.dependency_overrides.clear()
