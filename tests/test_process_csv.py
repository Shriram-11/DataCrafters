"""Tests for CSV process endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from settings import settings


def test_process_csv_memory_mode():
    client = TestClient(create_app())
    csv_content = b"name,age\nAlice,30\nBob,25\n"

    response = client.post(
        "/api/v1/process/csv",
        files={"file": ("people.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["storage_mode"] == "memory"
    assert data["rows"] == 2
    assert data["columns"] == 2


def test_process_csv_temp_mode_and_cleanup(monkeypatch):
    client = TestClient(create_app())
    monkeypatch.setattr(settings, "in_memory_threshold_mb", 0)

    temp_dir = Path(settings.temp_directory)
    before = set(temp_dir.glob("dc_upload_*.csv"))

    response = client.post(
        "/api/v1/process/csv",
        files={"file": ("large.csv", b"a,b\n1,2\n3,4\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["storage_mode"] == "temp_file"
    after = set(temp_dir.glob("dc_upload_*.csv"))
    assert after == before


def test_process_csv_rejects_non_csv():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/process/csv",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_process_csv_rejects_oversized(monkeypatch):
    client = TestClient(create_app())
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)

    response = client.post(
        "/api/v1/process/csv",
        files={"file": ("too_big.csv", b"a\n1\n", "text/csv")},
    )

    assert response.status_code == 413
