"""Tests for recommendations endpoint."""

from fastapi.testclient import TestClient

from app import create_app


def test_recommendations_csv_returns_column_recommendations():
    client = TestClient(create_app())
    csv_content = (
        b"age,income,city\n"
        b"21,1000,NY\n"
        b"33,,SF\n"
        b"25,1400,NY\n"
        b",2000,LA\n"
    )

    response = client.post(
        "/api/v1/recommendations/csv",
        files={"file": ("rec.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "column_recommendations" in body["data"]
    assert "age" in body["data"]["column_recommendations"]


def test_recommendations_csv_rejects_non_csv():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/recommendations/csv",
        files={"file": ("rec.txt", b"x", "text/plain")},
    )
    assert response.status_code == 400
