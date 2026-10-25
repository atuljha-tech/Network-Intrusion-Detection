"""
test_phase6.py — Unit and integration tests for Phase 6 (Dashboard & Alert Management).

Tests:
  1. AlertManager SQLite database initialization and schema creation.
  2. Single alert creation, retrieval, and fields validation.
  3. Batch alert ingestion from prediction DataFrame.
  4. Querying and filtering by date range, predicted class, status, confidence, anomaly flag.
  5. Alert lifecycle transitions and analyst notes updates.
  6. Rejection of invalid status transitions.
  7. Alert persistence across AlertManager restarts on the same SQLite file.
  8. Sanitized CSV report export and payload privacy verification.
  9. Input validation on CSV features schema (detecting missing columns and stripping target leakage columns).
  10. Alert deletion and database statistics aggregation.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.alert_manager import AlertManager
from src.config import ALERT_STATUS_CHOICES, DEFAULT_ALERT_STATUS
from dashboard.app import validate_csv_columns


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary SQLite database path for isolated testing."""
    return tmp_path / "test_alerts.db"


@pytest.fixture
def alert_mgr(temp_db_path):
    """Instantiate an AlertManager using a temporary database."""
    return AlertManager(temp_db_path)


# ── Alert Persistence Tests ───────────────────────────────────────────────────

def test_alert_manager_init(temp_db_path):
    """Test that SQLite database and alerts table are automatically initialized."""
    mgr = AlertManager(temp_db_path)
    assert temp_db_path.exists()
    stats = mgr.get_alert_statistics()
    assert stats["total_alerts"] == 0
    assert stats["unresolved_count"] == 0


def test_create_and_get_single_alert(alert_mgr):
    """Test creating a single alert and retrieving it by ID and query."""
    aid = alert_mgr.create_alert(
        predicted_class="dos hulk",
        model_version="RandomForest_v1",
        confidence_score=0.92,
        flow_id="192.168.1.100:443-10.0.0.5:54321-6",
        src_ip="192.168.1.100",
        src_port=443,
        dst_ip="10.0.0.5",
        dst_port=54321,
        protocol="TCP",
        anomaly_score=0.75,
        is_anomaly=True,
        status="New",
        analyst_notes="Initial automated detection",
        explanation="Flow bytes/sec deviation",
    )
    assert aid == 1

    # Retrieve by ID
    alert = alert_mgr.get_alert_by_id(aid)
    assert alert is not None
    assert alert["id"] == 1
    assert alert["predicted_class"] == "dos hulk"
    assert alert["confidence_score"] == pytest.approx(0.92)
    assert alert["anomaly_score"] == pytest.approx(0.75)
    assert alert["is_anomaly"] == 1
    assert alert["status"] == "New"
    assert alert["src_ip"] == "192.168.1.100"
    assert alert["dst_ip"] == "10.0.0.5"


def test_alert_persistence_across_restarts(temp_db_path):
    """Test that alerts persist on disk when a new AlertManager instance is created."""
    mgr1 = AlertManager(temp_db_path)
    aid = mgr1.create_alert(
        predicted_class="ddos",
        model_version="multiclass_v1",
        confidence_score=0.98,
        status="New",
    )
    assert aid > 0

    # Instantiate a second manager pointing to the same file
    mgr2 = AlertManager(temp_db_path)
    retrieved = mgr2.get_alert_by_id(aid)
    assert retrieved is not None
    assert retrieved["predicted_class"] == "ddos"
    assert retrieved["confidence_score"] == pytest.approx(0.98)


def test_update_alert_status_lifecycle(alert_mgr):
    """Test status transition from New to Investigating, then to Confirmed Incident."""
    aid = alert_mgr.create_alert(predicted_class="portscan", status="New")

    # Transition to Investigating
    assert alert_mgr.update_alert_status(aid, "Investigating", notes="Analyst reviewing SIEM logs.")
    a1 = alert_mgr.get_alert_by_id(aid)
    assert a1["status"] == "Investigating"
    assert "Analyst reviewing SIEM logs" in a1["analyst_notes"]

    # Transition to Confirmed Incident
    assert alert_mgr.update_alert_status(aid, "Confirmed Incident", notes="Reconnaissance scan confirmed.")
    a2 = alert_mgr.get_alert_by_id(aid)
    assert a2["status"] == "Confirmed Incident"

    # Transition to Resolved
    assert alert_mgr.update_alert_status(aid, "Resolved", notes="Firewall rule deployed.")
    a3 = alert_mgr.get_alert_by_id(aid)
    assert a3["status"] == "Resolved"


def test_invalid_status_rejection(alert_mgr):
    """Test that invalid statuses are rejected with ValueError."""
    aid = alert_mgr.create_alert(predicted_class="bot", status="New")
    with pytest.raises(ValueError):
        alert_mgr.update_alert_status(aid, "InvalidStatusXYZ")


def test_delete_and_clear_alerts(alert_mgr):
    """Test single alert deletion and bulk database clearing."""
    aid1 = alert_mgr.create_alert(predicted_class="bot")
    aid2 = alert_mgr.create_alert(predicted_class="ddos")
    assert alert_mgr.get_alert_statistics()["total_alerts"] == 2

    # Delete one
    assert alert_mgr.delete_alert(aid1) is True
    assert alert_mgr.get_alert_by_id(aid1) is None
    assert alert_mgr.get_alert_statistics()["total_alerts"] == 1

    # Clear all
    cleared = alert_mgr.clear_all_alerts()
    assert cleared == 1
    assert alert_mgr.get_alert_statistics()["total_alerts"] == 0


# ── Batch DataFrame Alert Ingestion Tests ─────────────────────────────────────

def test_create_alerts_from_df(alert_mgr):
    """Test bulk ingestion from a predictions DataFrame."""
    now_iso = datetime.now(timezone.utc).isoformat()
    df_preds = pd.DataFrame([
        {
            "flow_id": "flow_1",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 1234,
            "dst_port": 80,
            "protocol_name": "TCP",
            "predicted_class": "dos hulk",
            "score": 0.95,
            "anomaly_score": 0.65,
            "is_anomaly": True,
            "timestamp": now_iso,
            "explanation": "high flow bytes/sec",
        },
        {
            "flow_id": "flow_2",
            "src_ip": "10.0.0.3",
            "dst_ip": "10.0.0.4",
            "src_port": 5678,
            "dst_port": 443,
            "protocol_name": "TCP",
            "predicted_class": "benign",
            "score": 0.99,
            "anomaly_score": 0.10,
            "is_anomaly": False,
            "timestamp": now_iso,
        },
        {
            "flow_id": "flow_3",
            "src_ip": "10.0.0.5",
            "dst_ip": "10.0.0.6",
            "src_port": 9999,
            "dst_port": 22,
            "protocol_name": "TCP",
            "predicted_class": "ssh-patator",
            "score": 0.88,
            "anomaly_score": 0.55,
            "is_anomaly": False,
            "timestamp": now_iso,
        },
    ])

    # Filter malicious only (default) -> flow_2 (benign, non-anomaly) is omitted
    inserted = alert_mgr.create_alerts_from_df(df_preds, model_version="test_model", filter_malicious_only=True)
    assert inserted == 2

    alerts = alert_mgr.get_alerts()
    assert len(alerts) == 2
    classes = set(alerts["predicted_class"])
    assert "dos hulk" in classes
    assert "ssh-patator" in classes
    assert "benign" not in classes


def test_create_alerts_from_df_with_anomaly_only(alert_mgr):
    """Test that benign flows flagged as anomalies are preserved with anomaly label."""
    df_preds = pd.DataFrame([
        {
            "flow_id": "anomaly_benign_flow",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "label": "benign",
            "score": 0.70,
            "anomaly_score": 0.85,
            "is_anomaly": True,
        }
    ])
    inserted = alert_mgr.create_alerts_from_df(df_preds, filter_malicious_only=True)
    assert inserted == 1
    alerts = alert_mgr.get_alerts()
    assert "Anomaly" in alerts.iloc[0]["predicted_class"]
    assert alerts.iloc[0]["is_anomaly"] == 1


# ── Query & Filter Tests ──────────────────────────────────────────────────────

def test_query_filtering(alert_mgr):
    """Test filtering alerts by class, status, confidence, and search query."""
    base_time = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Insert diverse alerts
    alert_mgr.create_alert(
        predicted_class="ddos",
        status="New",
        confidence_score=0.95,
        src_ip="1.1.1.1",
        dst_ip="2.2.2.2",
        timestamp=base_time.isoformat(),
    )
    alert_mgr.create_alert(
        predicted_class="portscan",
        status="Investigating",
        confidence_score=0.60,
        src_ip="3.3.3.3",
        dst_ip="4.4.4.4",
        timestamp=(base_time + timedelta(days=1)).isoformat(),
    )
    alert_mgr.create_alert(
        predicted_class="ddos",
        status="Confirmed Incident",
        confidence_score=0.85,
        src_ip="5.5.5.5",
        dst_ip="6.6.6.6",
        timestamp=(base_time + timedelta(days=2)).isoformat(),
    )

    # Filter by class
    df_ddos = alert_mgr.get_alerts(predicted_class="ddos")
    assert len(df_ddos) == 2

    # Filter by status
    df_inv = alert_mgr.get_alerts(status="Investigating")
    assert len(df_inv) == 1
    assert df_inv.iloc[0]["predicted_class"] == "portscan"

    # Filter by min confidence
    df_high_conf = alert_mgr.get_alerts(min_confidence=0.90)
    assert len(df_high_conf) == 1
    assert df_high_conf.iloc[0]["confidence_score"] == pytest.approx(0.95)

    # Filter by search query
    df_search = alert_mgr.get_alerts(search_query="3.3.3.3")
    assert len(df_search) == 1
    assert df_search.iloc[0]["src_ip"] == "3.3.3.3"

    # Filter by date range
    start = (base_time + timedelta(hours=12)).isoformat()
    df_date = alert_mgr.get_alerts(start_date=start)
    assert len(df_date) == 2


# ── CSV Report Export & Safety Tests ──────────────────────────────────────────

def test_export_alerts_to_csv(alert_mgr, tmp_path):
    """Test sanitized CSV export without exposing raw payloads."""
    alert_mgr.create_alert(
        predicted_class="dos hulk",
        flow_id="test_flow_export",
        src_ip="172.16.0.1",
        dst_ip="192.168.1.1",
        confidence_score=0.94,
    )
    export_file = tmp_path / "exported_alerts.csv"
    res_path = alert_mgr.export_alerts_to_csv(export_file)

    assert res_path.exists()
    df_exported = pd.read_csv(res_path)
    assert len(df_exported) == 1
    assert "raw_payload" not in df_exported.columns
    assert "payload" not in df_exported.columns
    assert "src_ip" in df_exported.columns
    assert "predicted_class" in df_exported.columns
    assert df_exported.iloc[0]["flow_id"] == "test_flow_export"


# ── Feature Schema Validation Tests ───────────────────────────────────────────

def test_validate_csv_columns_success():
    """Test that a DataFrame with the 66 model features passes validation."""
    sample_csv = Path("data/sample_features.csv")
    if sample_csv.exists():
        df = pd.read_csv(sample_csv)
        is_valid, missing, stripped = validate_csv_columns(df)
        assert is_valid is True
        assert len(missing) == 0


def test_validate_csv_columns_missing_features():
    """Test that missing required columns are properly caught and flagged."""
    df_incomplete = pd.DataFrame({"col_a": [1.0], "col_b": [2.0]})
    is_valid, missing, stripped = validate_csv_columns(df_incomplete)
    assert is_valid is False
    assert len(missing) > 0


def test_validate_csv_columns_strips_target_leakage():
    """Test that target / metadata columns (label, is_malicious) are safely separated."""
    sample_csv = Path("data/sample_features.csv")
    if sample_csv.exists():
        df = pd.read_csv(sample_csv).copy()
        df["label"] = "benign"
        df["is_malicious"] = 0
        df["_source_file"] = "test.csv"

        is_valid, missing, stripped = validate_csv_columns(df)
        assert is_valid is True
        assert len(missing) == 0
        assert "label" in stripped
        assert "is_malicious" in stripped
        assert "_source_file" in stripped

