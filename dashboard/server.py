"""
server.py — High-Performance Local REST API & Web Server for AI-NIDS SOC Dashboard.

Built with Flask 3.x and Python standard library:
  • GET  /                       → Serves the single-page application (index.html).
  • GET  /api/health             → Instant health verification of model pipelines and SQLite store.
  • GET  /api/stats              → Executive telemetry KPIs and operational status.
  • GET  /api/alerts             → Filterable, paginated alert records.
  • GET  /api/alerts/<id>        → Detailed single alert record with XAI attributions.
  • POST /api/alerts/<id>/status → In-place alert status and analyst notes updater.
  • POST /api/ingest/csv         → Batch CSV feature validation and inference.
  • POST /api/ingest/sample-csv  → One-click evaluation of data/sample_features.csv.
  • POST /api/ingest/pcap        → Raw PCAP packet parsing, flow reconstruction, and inference.
  • POST /api/ingest/sample-pcap → One-click evaluation of sample_network_traffic.pcap.
  • GET  /api/chart-data         → Formatted telemetry data for forensic distribution charts.
  • GET  /api/export/csv         → Sanitized CSV report download (zero payload leakage).
  • POST /api/alerts/clear       → Clear local alert database.

Budget & Governance:
  • ₹0 cost, 100% offline local execution, zero cloud dependencies.
  • Strictly non-destructive passive traffic analysis.
  • Flow metadata only, zero packet payload serialization.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

from src.alert_manager import AlertManager
from src.config import (
    ALERT_STATUS_CHOICES,
    ALERTS_DB_PATH,
    DEFAULT_ALERT_STATUS,
    MODELS_DIR,
    PCAP_DIR,
    REPORTS_DIR,
    setup_logging,
)
from src.model_utils import load_metadata
from src.pcap_processor import process_pcap
from src.predict import UnifiedPredictor

logger = setup_logging()
log = logging.getLogger("ai_nids.server")

# ── Paths & Singletons ────────────────────────────────────────────────────────

DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)

# Backend singletons
alert_mgr = AlertManager(ALERTS_DB_PATH)
predictor = UnifiedPredictor(auto_load=True)


# ── Feature Schema Validation ────────────────────────────────────────────────

def validate_csv_columns(df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
    """Validate that DataFrame contains the required 66 CIC-IDS2017 features.

    Returns (is_valid, missing_columns, stripped_columns)
    """
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        return False, ["Model metadata not found on disk."], []

    meta = load_metadata(metadata_path)
    expected_features = set(meta["feature_names"])

    # Separate target/metadata columns to prevent data leakage
    stripped = []
    from src.config import NON_FEATURE_COLUMNS
    for col in list(df.columns):
        if col in NON_FEATURE_COLUMNS or col.lower() in ["label", "is_malicious", "class"]:
            stripped.append(col)

    remaining_cols = set(df.columns) - set(stripped)
    missing = sorted(list(expected_features - remaining_cols))
    return len(missing) == 0, missing, stripped


# ── CORS Headers Helper ───────────────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response


# ── Web Routes & Endpoints ────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the single-page dashboard HTML application."""
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/health", methods=["GET"])
def get_health():
    """Instant backend and model pipeline health verification."""
    binary_ok = predictor.binary_pipeline is not None
    multiclass_ok = predictor.multiclass_pipeline is not None
    anomaly_ok = predictor.anomaly_pipeline is not None
    db_ok = ALERTS_DB_PATH.exists()

    status_str = "OPTIMAL" if (binary_ok and multiclass_ok and anomaly_ok and db_ok) else "DEGRADED"
    return jsonify({
        "status": status_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "binary_model": "ACTIVE [RECALL-PRIORITIZED]" if binary_ok else "OFFLINE",
        "multiclass_model": "ACTIVE [13 ATTACK CATEGORIES]" if multiclass_ok else "OFFLINE",
        "anomaly_engine": "ACTIVE [ISOLATION FOREST]" if anomaly_ok else "OFFLINE",
        "storage": f"SQLITE LOCAL [{ALERTS_DB_PATH.name}]",
        "platform": "100% Offline Localhost Execution (₹0 Budget)",
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Executive KPI metric cards and telemetry summary."""
    stats = alert_mgr.get_alert_statistics()

    # Empirical benchmark latencies from Phase 7 benchmark
    avg_latency = 10.9  # ms/flow (multiclass median)
    throughput = 38240  # flows/sec

    perf_file = REPORTS_DIR / "performance_metrics.json"
    if perf_file.exists():
        try:
            with open(perf_file, "r", encoding="utf-8") as f:
                perf_data = json.load(f)
                avg_latency = round(perf_data.get("single_flow_latency_ms", {}).get("Multiclass Classifier", {}).get("median_ms", 10.9), 2)
                throughput = int(perf_data.get("batch_throughput_flows_sec", {}).get("Multiclass Classifier", {}).get("batch_1000", {}).get("throughput_flows_per_sec", 38240))
        except Exception:
            pass

    return jsonify({
        "total_alerts": stats["total_alerts"],
        "unresolved_count": stats["unresolved_count"],
        "confirmed_count": stats["confirmed_count"],
        "false_positive_count": stats["false_positive_count"],
        "anomaly_count": stats["anomaly_count"],
        "avg_latency_ms": avg_latency,
        "throughput_flows_sec": throughput,
        "by_status": stats["by_status"],
        "by_class": stats["by_class"],
        "system": {
            "binary_model": "Active [Recall-Prioritized]" if predictor.binary_pipeline else "Offline",
            "multiclass_model": "Active [13 Classes]" if predictor.multiclass_pipeline else "Offline",
            "anomaly_engine": "Calibrated [Isolation Forest]" if predictor.anomaly_pipeline else "Offline",
            "storage": f"SQLite Local [{ALERTS_DB_PATH}]",
        },
    })


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Retrieve filterable, paginated alert records from SQLite store."""
    search = request.args.get("search")
    status = request.args.get("status")
    predicted_class = request.args.get("predicted_class")
    min_confidence = request.args.get("min_confidence", type=float)
    is_anomaly_arg = request.args.get("is_anomaly")
    is_anomaly = (is_anomaly_arg.lower() in ("true", "1")) if is_anomaly_arg else None
    limit = request.args.get("limit", default=250, type=int)
    offset = request.args.get("offset", default=0, type=int)

    df_alerts = alert_mgr.get_alerts(
        search_query=search,
        status=status,
        predicted_class=predicted_class,
        min_confidence=min_confidence,
        is_anomaly=is_anomaly,
        limit=limit,
        offset=offset,
    )

    records = []
    for _, row in df_alerts.iterrows():
        rec = row.to_dict()
        expl_raw = rec.get("explanation", "")
        if isinstance(expl_raw, str) and (expl_raw.startswith("{") or expl_raw.startswith("[")):
            try:
                rec["explanation_parsed"] = json.loads(expl_raw)
            except Exception:
                rec["explanation_parsed"] = expl_raw
        else:
            rec["explanation_parsed"] = expl_raw
        records.append(rec)

    return jsonify({
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "alerts": records,
    })


@app.route("/api/alerts/<int:alert_id>", methods=["GET"])
def get_single_alert(alert_id: int):
    """Retrieve full forensic dossier for a single alert."""
    record = alert_mgr.get_alert_by_id(alert_id)
    if not record:
        return jsonify({"error": f"Alert #{alert_id} not found."}), 404

    expl_raw = record.get("explanation", "")
    if isinstance(expl_raw, str) and (expl_raw.startswith("{") or expl_raw.startswith("[")):
        try:
            record["explanation_parsed"] = json.loads(expl_raw)
        except Exception:
            record["explanation_parsed"] = expl_raw
    else:
        record["explanation_parsed"] = expl_raw

    return jsonify(record)


@app.route("/api/alerts/<int:alert_id>/status", methods=["POST"])
def update_alert_status(alert_id: int):
    """Update lifecycle status and forensic notes for an alert."""
    payload = request.get_json(silent=True) or {}
    new_status = payload.get("status")
    notes = payload.get("analyst_notes", "")

    if not new_status or new_status not in ALERT_STATUS_CHOICES:
        return jsonify({"error": f"Invalid status '{new_status}'. Allowed: {ALERT_STATUS_CHOICES}"}), 400

    updated = alert_mgr.update_alert_status(alert_id=alert_id, new_status=new_status, notes=notes)
    if not updated:
        return jsonify({"error": f"Alert #{alert_id} not found."}), 404

    refreshed = alert_mgr.get_alert_by_id(alert_id)
    return jsonify({"success": True, "alert": refreshed})


@app.route("/api/chart-data", methods=["GET"])
def get_chart_data():
    """Return aggregated telemetry data specifically formatted for frontend forensic charts."""
    stats = alert_mgr.get_alert_statistics()

    attack_classes = list(stats["by_class"].keys())
    attack_counts = list(stats["by_class"].values())

    statuses = ALERT_STATUS_CHOICES
    status_counts = [stats["by_status"].get(st, 0) for st in statuses]

    latency_models = ["Binary Model", "Multiclass", "Isolation Forest", "XAI Attribution"]
    latencies = [10.7, 10.9, 30.2, 14.1]

    return jsonify({
        "attack_breakdown": {
            "labels": attack_classes[:10],
            "values": attack_counts[:10],
        },
        "lifecycle_distribution": {
            "labels": statuses,
            "values": status_counts,
        },
        "performance_telemetry": {
            "labels": latency_models,
            "latencies_ms": latencies,
            "throughput_flows_sec": 38240,
        },
    })


@app.route("/api/ingest/csv", methods=["POST"])
def ingest_csv():
    """Batch-ingest an uploaded CSV file, validate 66 features, run inference, and persist alerts."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in form parameter 'file'."}), 400

    uploaded_file = request.files["file"]
    model_type = request.form.get("model_type", "multiclass")
    include_anomaly = request.form.get("include_anomaly", "true").lower() in ("true", "1")
    include_xai = request.form.get("include_xai", "true").lower() in ("true", "1")
    min_confidence = float(request.form.get("min_confidence", 0.0))

    try:
        df_input = pd.read_csv(uploaded_file.stream)
    except Exception as exc:
        return jsonify({"error": f"Invalid CSV file format: {exc}"}), 400

    is_valid, missing, stripped = validate_csv_columns(df_input)
    if not is_valid:
        return jsonify({
            "error": "Schema Validation Failed",
            "message": f"Missing {len(missing)} required features.",
            "missing_features": missing[:15],
            "expected_count": 66,
            "provided_columns": len(df_input.columns),
        }), 422

    clean_df = df_input.drop(columns=[c for c in stripped if c in df_input.columns])

    t0 = time.time()
    try:
        res_df = predictor.predict(
            clean_df,
            model_type=model_type,
            include_anomaly=include_anomaly,
            include_explanation=include_xai,
        )
    except Exception as exc:
        return jsonify({"error": f"Inference execution error: {exc}"}), 500

    elapsed_sec = round(time.time() - t0, 4)
    avg_latency_ms = round((elapsed_sec / max(1, len(clean_df))) * 1000.0, 2)

    alerts_added = alert_mgr.create_alerts_from_df(
        res_df,
        model_version=f"{model_type}_pipeline",
        min_confidence=min_confidence,
        filter_malicious_only=True,
    )

    lbl_col = "predicted_class" if "predicted_class" in res_df.columns else "label"
    benign_count = int(sum(res_df[lbl_col].astype(str).str.lower() == "benign"))
    threat_count = int(len(res_df) - benign_count)
    anomaly_count = int(sum(res_df["is_anomaly"] == 1)) if "is_anomaly" in res_df.columns else 0

    preview_cols = [c for c in [lbl_col, "score", "anomaly_score", "is_anomaly", "explanation"] if c in res_df.columns]
    preview_rows = res_df[preview_cols].head(15).to_dict(orient="records")

    return jsonify({
        "success": True,
        "rows_evaluated": len(df_input),
        "benign_count": benign_count,
        "threat_count": threat_count,
        "anomaly_count": anomaly_count,
        "alerts_registered": alerts_added,
        "processing_time_seconds": elapsed_sec,
        "avg_latency_ms": avg_latency_ms,
        "preview": preview_rows,
    })


@app.route("/api/ingest/sample-csv", methods=["POST"])
def ingest_sample_csv():
    """One-click evaluation of the bundled sample_features.csv."""
    sample_path = Path("data/sample_features.csv")
    if not sample_path.exists():
        return jsonify({"error": "Sample dataset data/sample_features.csv not found."}), 404

    model_type = request.form.get("model_type", "multiclass")
    include_anomaly = request.form.get("include_anomaly", "true").lower() in ("true", "1")
    include_xai = request.form.get("include_xai", "true").lower() in ("true", "1")
    min_confidence = float(request.form.get("min_confidence", 0.0))

    df_input = pd.read_csv(sample_path)
    is_valid, missing, stripped = validate_csv_columns(df_input)
    clean_df = df_input.drop(columns=[c for c in stripped if c in df_input.columns])

    t0 = time.time()
    res_df = predictor.predict(
        clean_df,
        model_type=model_type,
        include_anomaly=include_anomaly,
        include_explanation=include_xai,
    )
    elapsed_sec = round(time.time() - t0, 4)
    avg_latency_ms = round((elapsed_sec / max(1, len(clean_df))) * 1000.0, 2)

    alerts_added = alert_mgr.create_alerts_from_df(
        res_df,
        model_version=f"{model_type}_pipeline",
        min_confidence=min_confidence,
        filter_malicious_only=True,
    )

    lbl_col = "predicted_class" if "predicted_class" in res_df.columns else "label"
    benign_count = int(sum(res_df[lbl_col].astype(str).str.lower() == "benign"))
    threat_count = int(len(res_df) - benign_count)
    anomaly_count = int(sum(res_df["is_anomaly"] == 1)) if "is_anomaly" in res_df.columns else 0

    preview_cols = [c for c in [lbl_col, "score", "anomaly_score", "is_anomaly", "explanation"] if c in res_df.columns]
    preview_rows = res_df[preview_cols].head(15).to_dict(orient="records")

    return jsonify({
        "success": True,
        "source": "data/sample_features.csv",
        "rows_evaluated": len(df_input),
        "benign_count": benign_count,
        "threat_count": threat_count,
        "anomaly_count": anomaly_count,
        "alerts_registered": alerts_added,
        "processing_time_seconds": elapsed_sec,
        "avg_latency_ms": avg_latency_ms,
        "preview": preview_rows,
    })


@app.route("/api/ingest/pcap", methods=["POST"])
def ingest_pcap():
    """Dissect uploaded PCAP packet trace, reconstruct bidirectional flows, and run inference."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in form parameter 'file'."}), 400

    uploaded_file = request.files["file"]
    model_type = request.form.get("model_type", "multiclass")
    include_anomaly = request.form.get("include_anomaly", "true").lower() in ("true", "1")
    include_xai = request.form.get("include_xai", "true").lower() in ("true", "1")

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        df_flows, telemetry = process_pcap(
            tmp_path,
            predictor=predictor,
            model_type=model_type,
            include_anomaly=include_anomaly,
            include_explanation=include_xai,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    alerts_added = 0
    if not df_flows.empty:
        alerts_added = alert_mgr.create_alerts_from_df(
            df_flows,
            model_version=f"pcap_{model_type}_pipeline",
            min_confidence=0.0,
            filter_malicious_only=True,
        )

    view_cols = [
        c for c in [
            "flow_id", "src_ip", "src_port", "dst_ip", "dst_port", "protocol_name",
            "packet_count", "byte_count", "duration_ms", "predicted_class", "label",
            "score", "anomaly_score", "is_anomaly", "explanation"
        ] if c in df_flows.columns
    ]
    flow_previews = df_flows[view_cols].head(20).to_dict(orient="records") if not df_flows.empty else []

    return jsonify({
        "success": True,
        "telemetry": telemetry,
        "alerts_registered": alerts_added,
        "flows": flow_previews,
    })


@app.route("/api/ingest/sample-pcap", methods=["POST"])
def ingest_sample_pcap():
    """One-click evaluation of the bundled sample_network_traffic.pcap."""
    sample_pcap = PCAP_DIR / "sample_network_traffic.pcap"
    if not sample_pcap.exists():
        return jsonify({"error": "Sample PCAP file not found."}), 404

    model_type = request.form.get("model_type", "multiclass")
    include_anomaly = request.form.get("include_anomaly", "true").lower() in ("true", "1")
    include_xai = request.form.get("include_xai", "true").lower() in ("true", "1")

    df_flows, telemetry = process_pcap(
        sample_pcap,
        predictor=predictor,
        model_type=model_type,
        include_anomaly=include_anomaly,
        include_explanation=include_xai,
    )

    alerts_added = 0
    if not df_flows.empty:
        alerts_added = alert_mgr.create_alerts_from_df(
            df_flows,
            model_version=f"pcap_{model_type}_pipeline",
            min_confidence=0.0,
            filter_malicious_only=True,
        )

    view_cols = [
        c for c in [
            "flow_id", "src_ip", "src_port", "dst_ip", "dst_port", "protocol_name",
            "packet_count", "byte_count", "duration_ms", "predicted_class", "label",
            "score", "anomaly_score", "is_anomaly", "explanation"
        ] if c in df_flows.columns
    ]
    flow_previews = df_flows[view_cols].head(20).to_dict(orient="records") if not df_flows.empty else []

    return jsonify({
        "success": True,
        "source": "data/pcap/sample_network_traffic.pcap",
        "telemetry": telemetry,
        "alerts_registered": alerts_added,
        "flows": flow_previews,
    })


@app.route("/api/export/csv", methods=["GET"])
def export_alerts_csv():
    """Download sanitized compliance report of active alerts with zero payload data."""
    status = request.args.get("status")
    predicted_class = request.args.get("predicted_class")
    search = request.args.get("search")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_path = REPORTS_DIR / f"sanitized_soc_alerts_{timestamp_str}.csv"

    res_file = alert_mgr.export_alerts_to_csv(
        filepath=export_path,
        status=status if status != "All" else None,
        predicted_class=predicted_class if predicted_class != "All" else None,
        search_query=search,
    )

    if not res_file.exists():
        return jsonify({"error": "Failed to generate CSV export."}), 500

    return send_file(
        str(res_file),
        as_attachment=True,
        download_name=res_file.name,
        mimetype="text/csv",
    )


@app.route("/api/alerts/clear", methods=["POST"])
def clear_database():
    """Clear all records from local alert database."""
    cleared = alert_mgr.clear_all_alerts()
    return jsonify({"success": True, "cleared_count": cleared})


# ── Standalone CLI Runner ─────────────────────────────────────────────────────

def run_server(host: str = "127.0.0.1", port: int = 8501):
    """Launch the AI-NIDS SOC Dashboard via Flask development server."""
    print(f"Starting AI-NIDS SOC Dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()

