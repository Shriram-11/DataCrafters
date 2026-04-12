"""Tests for profile endpoint."""

from fastapi.testclient import TestClient

from app import create_app


def test_profile_csv_returns_expected_sections():
    client = TestClient(create_app())
    csv_content = b"age,score,city\n21,72,NY\n33,81,SF\n25,91,NY\n"

    response = client.post(
        "/api/v1/profile/csv",
        files={"file": ("profile.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True

    data = body["data"]
    assert "shape" in data
    assert data["shape"]["rows"] == 3
    assert "column_types" in data
    assert "null_count" in data
    assert "null_percent" in data
    assert "unique_count" in data
    assert "numeric_summary" in data


def test_profile_csv_rejects_non_csv():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/profile/csv",
        files={"file": ("profile.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
