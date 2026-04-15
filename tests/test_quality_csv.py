"""Tests for quality endpoint."""

from fastapi.testclient import TestClient

from app import create_app


def test_quality_csv_returns_expected_sections():
    client = TestClient(create_app())
    csv_content = b"age,score,city\n21,72,NY\n21,72,NY\n,81,SF\n25,91,NY\n"

    response = client.post(
        "/api/v2/analysis/quality/csv",
        files={"file": ("quality.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True

    data = body["data"]
    assert "duplicates_count" in data
    assert data["duplicates_count"] >= 1
    assert "missing_count" in data
    assert "completeness_score" in data
    assert "missing_by_column" in data
    assert "outlier_count_by_numeric_column" in data


def test_quality_csv_rejects_non_csv():
    client = TestClient(create_app())
    response = client.post(
        "/api/v2/analysis/quality/csv",
        files={"file": ("quality.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
