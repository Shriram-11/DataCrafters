"""Tests for migrated legacy analysis endpoint."""

from fastapi.testclient import TestClient

from app import create_app


def test_analysis_csv_returns_report_keys():
    client = TestClient(create_app())
    csv_content = (
        b"age,score,city\n"
        b"21,72,NY\n"
        b"33,81,SF\n"
        b"25,91,NY\n"
        b"45,65,LA\n"
        b"39,88,SF\n"
    )

    response = client.post(
        "/api/v1/analysis/csv",
        files={"file": ("sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "data" in body

    data = body["data"]
    expected_keys = {
        "dataset_overview",
        "descriptive_statistics",
        "data_quality_analysis",
        "outlier_analysis",
        "univariate_analysis",
        "bivariate_analysis",
        "clustering_analysis",
        "data_cleaning_suggestions",
        "feature_engineering_suggestions",
    }
    assert expected_keys.issubset(set(data.keys()))


def test_analysis_csv_rejects_non_csv():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/analysis/csv",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
