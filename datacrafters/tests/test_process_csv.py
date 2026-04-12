"""Tests for CSV process endpoint."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from datacrafters.app.main import create_app
from datacrafters.app.core.config import settings


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


def test_process_csv_memory_mode(client):
    """Small CSV should be processed in memory."""
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
    assert data["column_names"] == ["name", "age"]


def test_process_csv_temp_mode_and_cleanup(client, monkeypatch):
    """Large CSV should use temp file mode and clean temp files afterwards."""
    monkeypatch.setattr(settings, "in_memory_threshold_mb", 0)

    temp_dir = Path(settings.temp_directory)
    before = set(temp_dir.glob("dc_upload_*.csv"))

    csv_content = b"a,b\n1,2\n3,4\n"
    response = client.post(
        "/api/v1/process/csv",
        files={"file": ("large.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["storage_mode"] == "temp_file"

    after = set(temp_dir.glob("dc_upload_*.csv"))
    assert after == before


def test_process_csv_rejects_non_csv(client):
    """Non-CSV uploads should fail validation."""
    response = client.post(
        "/api/v1/process/csv",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Only .csv files are supported"


def test_process_csv_rejects_oversized(client, monkeypatch):
    """Uploads above configured max size should return 413."""
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)

    response = client.post(
        "/api/v1/process/csv",
        files={"file": ("too_big.csv", b"a\n1\n", "text/csv")},
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "Uploaded file exceeds maximum size"
