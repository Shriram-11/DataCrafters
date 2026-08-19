"""Tests for async jobs endpoints."""

import time

from fastapi.testclient import TestClient

from app import create_app


def test_enqueue_profile_job_and_fetch_status():
    client = TestClient(create_app())
    csv_content = b"age,score,city\n21,72,NY\n33,81,SF\n25,91,NY\n"

    enqueue_response = client.post(
        "/api/v2/jobs/profile/csv",
        files={"file": ("profile.csv", csv_content, "text/csv")},
    )

    assert enqueue_response.status_code == 202
    enqueue_body = enqueue_response.json()
    assert enqueue_body["success"] is True
    job_id = enqueue_body["data"]["job_id"]

    body = {}
    for _ in range(20):
        status_response = client.get(f"/api/v2/jobs/{job_id}")
        assert status_response.status_code == 200
        body = status_response.json()
        if body["data"]["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert body["data"]["status"] == "completed"
    assert body["data"]["result"] is not None
    assert "shape" in body["data"]["result"]


def test_get_job_returns_404_when_job_missing():
    client = TestClient(create_app())
    response = client.get("/api/v2/jobs/unknown-job-id")

    assert response.status_code == 404
