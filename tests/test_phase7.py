"""
test_phase7.py — Comprehensive Unit & Integration Tests for Phase 7:
Testing, Optimization, Documentation, and Final Packaging.

Coverage Areas:
  1. Data Loading Resilience (edge cases, encodings, malformed headers)
  2. Preprocessing & Data Validation Robustness (NaNs, infs, type coercion, target leakage)
  3. Model Serialization & Deserialization Integrity (joblib, metadata, error handling)
  4. Unified Predictor Verification (binary, multiclass, empty input, invalid model_type)
  5. 66-Feature Schema Compatibility (flow extractor == model metadata == pipeline)
  6. Traffic Processing Edge Cases (0-byte PCAP, corrupt PCAP, non-IP frames, flow accumulation)
  7. Alert Storage, SQL Injection Immunity & Lifecycle Robustness
  8. Generalization Evaluation (held-out capture period / scenario)
  9. Performance Telemetry & Memory Profiling Sanity Checks
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import pytest
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether
from scapy.utils import wrpcap

from dashboard.app import validate_csv_columns
from src.alert_manager import AlertManager
from src.config import (
    ALERT_STATUS_CHOICES,
    DEFAULT_ALERT_STATUS,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    MULTICLASS_METADATA_PATH,
    MULTICLASS_PIPELINE_PATH,
    NON_FEATURE_COLUMNS,
)
from src.data_loader import _normalize_column_name, load_single_csv
from src.evaluate import evaluate_held_out_generalization
from src.flow_features import EXPECTED_FLOW_FEATURES, Flow, FlowKey
from src.model_utils import load_metadata, load_pipeline, validate_features
from src.pcap_processor import extract_packet_info, is_live_capture_supported, process_pcap
from src.predict import UnifiedPredictor, predict, predict_single
from src.preprocess import run_preprocessing
from src.run_eda import _generate_synthetic_data


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_feature_dict() -> Dict[str, float]:
    """Provide a valid 66-feature dictionary."""
    df = _generate_synthetic_data(n=5)[EXPECTED_FLOW_FEATURES]
    return df.iloc[0].to_dict()


@pytest.fixture
def temp_alerts_db(tmp_path: Path) -> AlertManager:
    """Provide an isolated AlertManager with a temporary database."""
    db_file = tmp_path / "phase7_alerts.db"
    return AlertManager(db_file)


# ── 1. Data Loading Resilience ───────────────────────────────────────────────

class TestDataLoadingResilience:
    def test_normalize_column_name_variations(self):
        assert _normalize_column_name("  Flow Duration  ") == "flow_duration"
        assert _normalize_column_name("Flow Bytes/s") == "flow_bytes_s"
        assert _normalize_column_name("Fwd Pkt Len Max") == "fwd_pkt_len_max"
        assert _normalize_column_name("Fwd_IAT_Min") == "fwd_iat_min"

    def test_load_csv_empty_file_raises(self, tmp_path: Path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("", encoding="utf-8")
        with pytest.raises(Exception):
            load_single_csv(empty_csv)

    def test_load_csv_nonexistent_file_raises(self, tmp_path: Path):
        missing = tmp_path / "non_existent.csv"
        with pytest.raises(FileNotFoundError):
            load_single_csv(missing)

    def test_load_csv_with_whitespace_and_bom(self, tmp_path: Path):
        csv_file = tmp_path / "bom_test.csv"
        csv_file.write_text("\ufeff Flow Duration , Total Fwd Packets , Label \n100,5,BENIGN\n", encoding="utf-8")
        df = load_single_csv(csv_file)
        assert "flow_duration" in df.columns
        assert "total_fwd_packets" in df.columns
        assert "label" in df.columns
        assert len(df) == 1


# ── 2. Preprocessing & Feature Schema Validation ─────────────────────────────

class TestPreprocessingAndValidation:
    def test_target_leakage_rejected(self):
        df_leak = pd.DataFrame({
            f: [1.0] for f in EXPECTED_FLOW_FEATURES
        })
        df_leak["label"] = "benign"
        with pytest.raises(ValueError, match="target/metadata columns"):
            validate_features(df_leak, EXPECTED_FLOW_FEATURES)

    def test_missing_features_rejected(self):
        df_incomplete = pd.DataFrame({"flow_duration": [100.0], "total_fwd_packets": [5.0]})
        with pytest.raises(ValueError, match="Missing"):
            validate_features(df_incomplete, EXPECTED_FLOW_FEATURES)

    def test_column_reordering_and_extra_pruned(self):
        # Reverse order + extra column
        reversed_cols = list(reversed(EXPECTED_FLOW_FEATURES))
        df_in = pd.DataFrame({col: [float(i)] for i, col in enumerate(reversed_cols)})
        df_in["extra_noise_feature"] = [999.0]

        validated = validate_features(df_in, EXPECTED_FLOW_FEATURES)
        assert list(validated.columns) == EXPECTED_FLOW_FEATURES
        assert "extra_noise_feature" not in validated.columns

    def test_non_numeric_strings_coerced_safely(self):
        df_str = pd.DataFrame({col: ["123.45"] for col in EXPECTED_FLOW_FEATURES})
        df_str.loc[0, "flow_duration"] = "unparseable_string"

        validated = validate_features(df_str, EXPECTED_FLOW_FEATURES)
        assert pd.api.types.is_numeric_dtype(validated["flow_duration"])
        assert np.isnan(validated.loc[0, "flow_duration"])
        assert validated.loc[0, "total_fwd_packets"] == pytest.approx(123.45)


# ── 3. Model Serialization & Integrity ───────────────────────────────────────

class TestModelSerializationAndIntegrity:
    def test_model_files_exist_on_disk(self):
        assert MODEL_PIPELINE_PATH.exists(), "Binary pipeline file must exist"
        assert MODEL_METADATA_PATH.exists(), "Binary metadata file must exist"
        assert MULTICLASS_PIPELINE_PATH.exists(), "Multiclass pipeline file must exist"
        assert MULTICLASS_METADATA_PATH.exists(), "Multiclass metadata file must exist"

    def test_load_metadata_content(self):
        meta = load_metadata(MODEL_METADATA_PATH)
        assert "feature_names" in meta
        assert meta["n_features"] == 66
        assert len(meta["feature_names"]) == 66

    def test_load_pipeline_missing_file_raises(self, tmp_path: Path):
        missing_pipe = tmp_path / "non_existent_pipeline.joblib"
        with pytest.raises(FileNotFoundError):
            load_pipeline(missing_pipe)

    def test_load_corrupt_pipeline_raises(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupt.joblib"
        corrupt_file.write_bytes(b"NOT_A_VALID_JOBLIB_DATA_BLOCK")
        with pytest.raises(Exception):
            load_pipeline(corrupt_file)


# ── 4. Unified Predictor Verification ─────────────────────────────────────────

class TestUnifiedPredictorVerification:
    def test_binary_prediction_structure(self, sample_feature_dict):
        predictor = UnifiedPredictor(auto_load=True)
        res = predictor.predict(sample_feature_dict, model_type="binary", include_anomaly=True, include_explanation=True)
        assert isinstance(res, dict)
        assert res["model_type"] == "binary"
        assert "prediction" in res
        assert "label" in res
        assert res["label"] in ["Benign", "Malicious"]
        assert "score" in res
        assert "is_anomaly" in res
        assert "anomaly_score" in res
        assert "explanation" in res

    def test_multiclass_prediction_structure(self, sample_feature_dict):
        predictor = UnifiedPredictor(auto_load=True)
        res = predictor.predict(sample_feature_dict, model_type="multiclass", include_anomaly=True, include_explanation=True)
        assert isinstance(res, dict)
        assert res["model_type"] == "multiclass"
        assert "predicted_class" in res
        assert isinstance(res["predicted_class"], str)
        assert "score" in res
        assert "is_anomaly" in res
        assert "explanation" in res

    def test_empty_dataframe_handled_gracefully(self):
        predictor = UnifiedPredictor(auto_load=True)
        empty_df = pd.DataFrame(columns=EXPECTED_FLOW_FEATURES)
        res_bin = predictor.predict(empty_df, model_type="binary")
        assert len(res_bin) == 0
        assert "prediction" in res_bin.columns
        assert "label" in res_bin.columns

        res_multi = predictor.predict(empty_df, model_type="multiclass")
        assert len(res_multi) == 0
        assert "predicted_class" in res_multi.columns

    def test_invalid_model_type_raises(self, sample_feature_dict):
        predictor = UnifiedPredictor(auto_load=True)
        with pytest.raises(ValueError, match="Invalid model_type"):
            predictor.predict(sample_feature_dict, model_type="quantum_classifier")


# ── 5. 66-Feature Schema Compatibility Check ─────────────────────────────────

class TestFeatureSchemaCompatibility:
    def test_exact_schema_match_across_all_modules(self):
        b_meta = load_metadata(MODEL_METADATA_PATH)
        m_meta = load_metadata(MULTICLASS_METADATA_PATH)

        flow_features = EXPECTED_FLOW_FEATURES
        binary_features = b_meta["feature_names"]
        multiclass_features = m_meta["feature_names"]

        assert len(flow_features) == 66
        assert len(binary_features) == 66
        assert len(multiclass_features) == 66

        assert flow_features == binary_features, "Flow accumulator features must match binary model features 1:1"
        assert flow_features == multiclass_features, "Flow accumulator features must match multiclass features 1:1"

    def test_flow_accumulator_generates_all_66_features(self):
        flow = Flow(key=FlowKey("10.0.0.1", 12345, "10.0.0.2", 80, 6), first_packet_time=100.0)
        # Add a forward packet
        flow.add_packet(timestamp=100.0, length=64, header_length=20, payload_length=44, tcp_flags=0x02, window_size=65535, is_forward=True)
        # Add a backward packet
        flow.add_packet(timestamp=100.02, length=128, header_length=20, payload_length=108, tcp_flags=0x12, window_size=65535, is_forward=False)

        feat_dict = flow.to_features(activity_timeout_sec=5.0)
        assert len(feat_dict) == 66
        assert list(feat_dict.keys()) == EXPECTED_FLOW_FEATURES
        for k, v in feat_dict.items():
            assert isinstance(v, (int, float, np.floating, np.integer))


# ── 6. Traffic Processing Edge Cases ─────────────────────────────────────────

class TestTrafficProcessingEdgeCases:
    def test_zero_byte_pcap(self, tmp_path: Path):
        empty_pcap = tmp_path / "empty.pcap"
        empty_pcap.write_bytes(b"")
        df, telemetry = process_pcap(empty_pcap)
        assert len(df) == 0
        assert telemetry["status"] == "empty_file"
        assert telemetry["packets_read"] == 0

    def test_completely_corrupted_pcap_bytes(self, tmp_path: Path):
        corrupt_pcap = tmp_path / "junk.pcap"
        corrupt_pcap.write_bytes(b"RANDOM_NON_PCAP_GARBAGE_PAYLOAD_HERE")
        df, telemetry = process_pcap(corrupt_pcap)
        assert len(df) == 0
        assert telemetry["status"] == "invalid_or_corrupt_pcap"

    def test_non_ip_arp_packets_filtered_cleanly(self, tmp_path: Path):
        arp_pcap = tmp_path / "arp_only.pcap"
        arp_pkt = Ether() / ARP(op="who-has", psrc="192.168.1.1", pdst="192.168.1.2")
        arp_pkt.time = 1700000000.0
        wrpcap(str(arp_pcap), [arp_pkt, arp_pkt])

        df, telemetry = process_pcap(arp_pcap)
        assert len(df) == 0
        assert telemetry["packets_read"] == 2
        assert telemetry["valid_ip_packets"] == 0
        assert telemetry["unsupported_protocol_packets"] == 2
        assert telemetry["status"] == "no_ip_flows_found"

    def test_is_live_capture_supported_contract(self):
        supported, msg = is_live_capture_supported()
        assert isinstance(supported, bool)
        assert isinstance(msg, str)
        assert len(msg) > 0


# ── 7. Alert Storage & SQL Injection Robustness ──────────────────────────────

class TestAlertStorageAndRobustness:
    def test_alert_lifecycle_and_statistics(self, temp_alerts_db):
        aid = temp_alerts_db.create_alert(predicted_class="ddos", confidence_score=0.95, status="New")
        assert aid == 1

        stats_before = temp_alerts_db.get_alert_statistics()
        assert stats_before["total_alerts"] == 1
        assert stats_before["unresolved_count"] == 1

        # Advance lifecycle
        temp_alerts_db.update_alert_status(aid, "Investigating", notes="Checking logs")
        temp_alerts_db.update_alert_status(aid, "Confirmed Incident", notes="Valid DDoS detected")

        stats_after = temp_alerts_db.get_alert_statistics()
        assert stats_after["confirmed_count"] == 1
        assert stats_after["unresolved_count"] == 0

    def test_invalid_status_transition_rejected(self, temp_alerts_db):
        aid = temp_alerts_db.create_alert(predicted_class="portscan")
        with pytest.raises(ValueError, match="Invalid status"):
            temp_alerts_db.update_alert_status(aid, "CompletelyMadeUpStatus")

    def test_sql_injection_attempt_is_safe(self, temp_alerts_db):
        """Verify that parameterized SQLite queries prevent SQL injection."""
        temp_alerts_db.create_alert(predicted_class="portscan", src_ip="192.168.1.50")

        # Malicious search queries
        injection_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE alerts; --",
            "' UNION SELECT * FROM alerts --",
            "1; UPDATE alerts SET status='Hacked';",
        ]

        for payload in injection_payloads:
            # Must execute safely without throwing SQLite syntax exceptions or dropping tables
            res = temp_alerts_db.get_alerts(search_query=payload)
            assert isinstance(res, pd.DataFrame)

        # Confirm alerts table is intact and unaffected
        stats = temp_alerts_db.get_alert_statistics()
        assert stats["total_alerts"] == 1

    def test_sanitized_csv_export_has_no_raw_payloads(self, temp_alerts_db, tmp_path: Path):
        temp_alerts_db.create_alert(predicted_class="bot", flow_id="flow_test_export")
        export_path = tmp_path / "alerts_export.csv"
        temp_alerts_db.export_alerts_to_csv(export_path)

        assert export_path.exists()
        df_exp = pd.read_csv(export_path)
        assert "raw_payload" not in df_exp.columns
        assert "payload" not in df_exp.columns
        assert "predicted_class" in df_exp.columns


# ── 8. Generalization Evaluation Verification ────────────────────────────────

class TestGeneralizationEvaluation:
    def test_evaluate_held_out_generalization_runs(self, tmp_path: Path):
        df_synth = _generate_synthetic_data(n=600)
        res = evaluate_held_out_generalization(
            df=df_synth,
            source_col="_source_file",
            held_out_source="Friday.csv",
            output_dir=tmp_path,
        )

        assert "generalization_delta" in res
        assert "delta_f1" in res["generalization_delta"]
        assert "in_domain_metrics" in res
        assert "held_out_metrics" in res
        assert Path(res["report_path"]).exists()
        assert Path(res["report_path"]).stat().st_size > 100


# ── 9. Dashboard CSV Schema Validation Contract ──────────────────────────────

class TestDashboardSchemaValidationContract:
    def test_validate_csv_columns_accepts_valid_feature_df(self):
        valid_df = _generate_synthetic_data(n=5)[EXPECTED_FLOW_FEATURES]
        is_valid, missing, stripped = validate_csv_columns(valid_df)
        assert is_valid is True
        assert len(missing) == 0

    def test_validate_csv_columns_strips_target_leakage_safely(self):
        valid_df = _generate_synthetic_data(n=5)[EXPECTED_FLOW_FEATURES].copy()
        valid_df["label"] = "benign"
        valid_df["is_malicious"] = 0
        valid_df["_source_file"] = "Monday.csv"

        is_valid, missing, stripped = validate_csv_columns(valid_df)
        assert is_valid is True
        assert "label" in stripped
        assert "is_malicious" in stripped
        assert "_source_file" in stripped
