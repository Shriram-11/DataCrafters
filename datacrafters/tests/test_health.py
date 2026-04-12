"""Tests for health endpoint."""
import pytest
from fastapi.testclient import TestClient

from datacrafters.app.main import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    """Test health endpoint returns 200 with expected schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "datacrafters"
    assert data["version"] == "0.1.0"
    assert "temp_dir_size_mb" in data
    assert "max_upload_size_mb" in data


def test_health_response_schema(client):
    """Test health response contains all required fields."""
    response = client.get("/api/v1/health")
    data = response.json()

    required_fields = {
        "status",
        "app_name",
        "version",
        "debug",
        "temp_dir_size_mb",
        "max_upload_size_mb",
    }
    assert set(data.keys()) == required_fields
