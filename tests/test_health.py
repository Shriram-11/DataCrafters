"""Tests for health endpoint."""

from fastapi.testclient import TestClient

from app import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "DataCrafters"
    assert data["version"] == "0.1.0"
    assert "temp_dir_size_mb" in data
    assert "max_upload_size_mb" in data
