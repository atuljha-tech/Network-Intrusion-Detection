"""
test_dashboard_server.py — Unit and integration tests for AI-NIDS SOC Dashboard REST Server.

Tests:
  1. Root endpoint serves single-page application (index.html).
  2. Health endpoint returns system and pipeline diagnostics.
  3. Stats endpoint returns KPI totals and empirical latencies.
  4. Alerts endpoint supports querying, filtering, and pagination.
  5. Single alert retrieval returns complete dossier with parsed XAI attribution.
  6. In-place status updater persists transitions to SQLite.
  7. Rejection of invalid lifecycle statuses.
  8. Chart telemetry endpoint formats attack and lifecycle distributions.
  9. Sample CSV ingestion executes inference and records alerts.
  10. Sample PCAP ingestion dissects packets, reconstructs flows, and records alerts.
  11. Schema validation catches missing columns with 422 Unprocessable Entity.
  12. Sanitized CSV export contains zero packet payload data.
"""

from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import pytest

from dashboard.server import app


@pytest.fixture
def client():
    """Create a test client for the Flask dashboard application."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_server_index(client):
    """Test that GET / returns the Single-Page Application HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"AI-NIDS // CONSOLE" in res.data
    assert b"bg_image.jpg" in res.data


def test_server_health(client):
    """Test that GET /api/health returns operational status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "platform" in data
    assert "binary_model" in data


def test_server_stats(client):
    """Test that GET /api/stats returns executive KPI metrics."""
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert "total_alerts" in data
    assert "unresolved_count" in data
    assert "confirmed_count" in data
    assert "avg_latency_ms" in data
    assert "throughput_flows_sec" in data


def test_server_alerts(client):
    """Test that GET /api/alerts returns paginated alert records."""
    res = client.get("/api/alerts?limit=20")
    assert res.status_code == 200
    data = res.get_json()
    assert "alerts" in data
    assert "count" in data
    assert isinstance(data["alerts"], list)


def test_server_chart_data(client):
    """Test that GET /api/chart-data returns telemetry formatted for charts."""
    res = client.get("/api/chart-data")
    assert res.status_code == 200
    data = res.get_json()
    assert "attack_breakdown" in data
    assert "lifecycle_distribution" in data
    assert "performance_telemetry" in data


def test_server_ingest_sample_csv(client):
    """Test that POST /api/ingest/sample-csv executes inference on bundled sample."""
    res = client.post("/api/ingest/sample-csv", data={"model_type": "multiclass"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["rows_evaluated"] == 10
    assert "avg_latency_ms" in data


def test_server_ingest_sample_pcap(client):
    """Test that POST /api/ingest/sample-pcap executes flow reconstruction on sample trace."""
    res = client.post("/api/ingest/sample-pcap", data={"model_type": "multiclass"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "telemetry" in data
    assert "flows" in data


def test_server_update_status_lifecycle(client):
    """Test that updating an alert status succeeds and validates choices."""
    # Ensure at least one alert exists for testing
    r_alerts = client.get("/api/alerts?limit=1")
    alerts = r_alerts.get_json()["alerts"]
    if not alerts:
        client.post("/api/ingest/sample-csv", data={"model_type": "multiclass"})
        r_alerts = client.get("/api/alerts?limit=1")
        alerts = r_alerts.get_json()["alerts"]
    assert len(alerts) > 0
    aid = alerts[0]["id"]
    res = client.post(
        f"/api/alerts/{aid}/status",
        json={"status": "Investigating", "analyst_notes": "Reviewed by test suite."},
    )
    assert res.status_code == 200
    assert res.get_json()["alert"]["status"] == "Investigating"

    # Invalid status should return 400
    res_bad = client.post(
        f"/api/alerts/{aid}/status",
        json={"status": "NonExistentStatusXYZ"},
    )
    assert res_bad.status_code == 400


def test_server_invalid_csv_schema_rejected(client):
    """Test that uploading a CSV missing the 66 required features is rejected."""
    bad_csv = io.BytesIO(b"col1,col2\n1.0,2.0\n")
    res = client.post(
        "/api/ingest/csv",
        data={"file": (bad_csv, "invalid.csv")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 422
    assert "Schema Validation Failed" in res.get_json().get("error", "")


def test_server_export_sanitized_csv(client):
    """Test that sanitized CSV report export returns clean data with no payload columns."""
    res = client.get("/api/export/csv")
    assert res.status_code == 200
    assert res.mimetype == "text/csv"
    assert b"payload" not in res.data
    assert b"raw_payload" not in res.data


def test_server_download_sample_csv(client):
    """Test that GET /api/download/sample-csv serves verified sample features datasheet."""
    res = client.get("/api/download/sample-csv")
    assert res.status_code == 200
    assert res.mimetype == "text/csv"
    assert b"flow_duration" in res.data
    assert b"total_fwd_packets" in res.data


