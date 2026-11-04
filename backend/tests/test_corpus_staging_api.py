from fastapi.testclient import TestClient

from app.contracts import UserProfile
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
        assert response.json()["schema_version"] == "dmbok-structure-v1"
        assert response.json()["entries"][0]["page_start"] == 1
    finally:
        app.dependency_overrides.clear()
