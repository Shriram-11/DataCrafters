"""Tests for correlation endpoint."""

from fastapi.testclient import TestClient

from app import create_app


def test_correlation_csv_returns_matrix_and_pairs():
    client = TestClient(create_app())
    csv_content = b"x,y,z\n1,2,3\n2,4,6\n3,6,9\n"

    response = client.post(
        "/api/v1/correlation/csv",
        files={"file": ("corr.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "columns" in body["data"]
    assert "matrix" in body["data"]
    assert "top_correlations" in body["data"]
    assert len(body["data"]["columns"]) == 3


def test_correlation_csv_rejects_non_csv():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/correlation/csv",
        files={"file": ("corr.txt", b"x", "text/plain")},
    )
    assert response.status_code == 400
