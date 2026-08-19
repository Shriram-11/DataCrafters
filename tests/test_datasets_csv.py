"""Tests for dataset metadata and versioning endpoints."""

from fastapi.testclient import TestClient

from app import create_app


def test_upload_dataset_and_list_and_detail():
    client = TestClient(create_app())
    csv_content = b"id,amount,city\n1,100,NY\n2,200,SF\n"

    upload_response = client.post(
        "/api/v2/datasets/upload",
        data={"dataset_name": "sales_data"},
        files={"file": ("sales.csv", csv_content, "text/csv")},
    )
    assert upload_response.status_code == 200
    upload_body = upload_response.json()
    assert upload_body["success"] is True
    dataset_id = upload_body["data"]["dataset_id"]
    assert upload_body["data"]["name"] == "sales_data"
    assert upload_body["data"]["latest_version"]["version_number"] == 1

    list_response = client.get("/api/v2/datasets")
    assert list_response.status_code == 200
    list_body = list_response.json()
    assert list_body["success"] is True
    assert any(d["dataset_id"] == dataset_id for d in list_body["data"])

    detail_response = client.get(f"/api/v2/datasets/{dataset_id}")
    assert detail_response.status_code == 200
    detail_body = detail_response.json()
    assert detail_body["success"] is True
    assert detail_body["data"]["dataset_id"] == dataset_id


def test_upload_dataset_version_and_list_versions():
    client = TestClient(create_app())
    v1 = b"id,amount\n1,100\n2,150\n"
    v2 = b"id,amount\n1,120\n2,180\n3,210\n"

    upload_v1 = client.post(
        "/api/v2/datasets/upload",
        files={"file": ("metrics.csv", v1, "text/csv")},
    )
    assert upload_v1.status_code == 200
    dataset_id = upload_v1.json()["data"]["dataset_id"]

    upload_v2 = client.post(
        f"/api/v2/datasets/{dataset_id}/versions/upload",
        files={"file": ("metrics_v2.csv", v2, "text/csv")},
    )
    assert upload_v2.status_code == 200
    assert upload_v2.json()["data"]["latest_version"]["version_number"] == 2

    versions_response = client.get(f"/api/v2/datasets/{dataset_id}/versions")
    assert versions_response.status_code == 200
    versions = versions_response.json()["data"]
    assert len(versions) == 2
    assert versions[0]["version_number"] == 2
    assert versions[1]["version_number"] == 1
