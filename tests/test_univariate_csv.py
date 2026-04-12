"""Tests for univariate endpoint."""

from fastapi.testclient import TestClient

from app import create_app


def test_univariate_csv_returns_numeric_and_categorical_stats():
    client = TestClient(create_app())
    csv_content = b"age,score,city\n21,72,NY\n33,81,SF\n25,91,NY\n"

    response = client.post(
        "/api/v1/univariate/csv",
        files={"file": ("uni.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "numeric" in body["data"]
    assert "categorical" in body["data"]
    assert "age" in body["data"]["numeric"]
    assert "city" in body["data"]["categorical"]


def test_univariate_csv_rejects_non_csv():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/univariate/csv",
        files={"file": ("uni.txt", b"x", "text/plain")},
    )
    assert response.status_code == 400
