"""
app.py — AI-NIDS SOC Dashboard and Alert Management Application.

Built with:
  • Streamlit: Interactive local SOC analyst web interface.
  • Pandas & Matplotlib: High-resolution telemetry charts and metrics.
  • SQLite: Persistent local alert repository.
  • UnifiedPredictor & PCAP Processor: Real-time ML inference and flow reconstruction.

Budget & Governance:
  • ₹0 budget, 100% offline local execution, zero paid cloud APIs.
  • Passive non-destructive network monitoring.
  • Strict privacy: flow metadata only, zero packet payload exposure.
  • Rigorous operational distinction between statistical threat hypotheses and confirmed incidents.
  • Zero cheesy icons or emojis — clean, authoritative cybersecurity typography.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.alert_manager import AlertManager
from src.config import (
    ALERT_STATUS_CHOICES,
    ALERTS_DB_PATH,
    DEFAULT_ALERT_STATUS,
    MODELS_DIR,
    PCAP_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    setup_logging,
)
from src.model_utils import load_metadata, validate_features
from src.pcap_processor import is_live_capture_supported, process_pcap
from src.predict import UnifiedPredictor

logger = setup_logging()
log = logging.getLogger("ai_nids.dashboard")

# Set Matplotlib headless backend
plt.switch_backend("Agg")


# ── Cached Resource Loaders ───────────────────────────────────────────────────

@st.cache_resource
def get_alert_manager() -> AlertManager:
    """Initialize and cache the AlertManager instance."""
    return AlertManager(ALERTS_DB_PATH)


@st.cache_resource
def get_unified_predictor() -> UnifiedPredictor:
    """Initialize and cache the UnifiedPredictor instance."""
    return UnifiedPredictor(auto_load=True)


def get_hero_bg_base64() -> str:
    """Read bg_image.jpg and encode as base64 for hero background."""
    img_path = PROJECT_ROOT / "bg_image.jpg"
    if img_path.exists():
        encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    return ""


# ── Helper Utilities ──────────────────────────────────────────────────────────

def render_compliance_notice():
    """Render prominent SOC operational distinction and privacy disclaimer without emojis."""
    st.markdown(
        """
        <div class="soc-warning-box">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;">
                OPERATIONAL BOUNDARY NOTICE // PREDICTED THREAT VS CONFIRMED INCIDENT
            </div>
            <div style="font-size: 12px; line-height: 1.5; color: #cbd5e1;">
                All machine learning classifications and anomaly scores are <strong>probabilistic threat hypotheses</strong>. They do NOT constitute confirmed security breaches until verified and escalated by a human security analyst. Alerts default to status <code>New</code> until triaged.
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #64748b; margin-top: 6px; text-transform: uppercase;">
                PRIVACY GUARANTEE: STATISTICAL FLOW METADATA ONLY (5-TUPLES, PACKET COUNTS, TIMING) // ZERO PAYLOADS OR CREDENTIALS RECORDED
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_csv_columns(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Validate that uploaded DataFrame contains the required 66 features.

    Returns
    -------
    (is_valid, missing_columns, stripped_columns)
    """
    metadata_path = MODELS_DIR / "model_metadata.json"
    if metadata_path.exists():
        meta = load_metadata(metadata_path)
        expected_features = set(meta["feature_names"])
    else:
        from src.flow_features import EXPECTED_FLOW_FEATURES
        expected_features = set(EXPECTED_FLOW_FEATURES)

    # Identify and strip non-feature columns (e.g. labels, filenames) to prevent leakage
    stripped = []
    from src.config import NON_FEATURE_COLUMNS
    for col in list(df.columns):
        if col in NON_FEATURE_COLUMNS or col.lower() in ["label", "is_malicious", "class"]:
            stripped.append(col)

    remaining_cols = set(df.columns) - set(stripped)
    missing = sorted(list(expected_features - remaining_cols))

    is_valid = len(missing) == 0
    return is_valid, missing, stripped


# ── Main Dashboard Application ────────────────────────────────────────────────

def main():
    """Render the AI-NIDS SOC Dashboard and Alert Management Application."""
    st.set_page_config(
        page_title="AI-NIDS Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    hero_bg = get_hero_bg_base64()
    bg_css = f"url('{hero_bg}')" if hero_bg else "none"

    # Custom Styling for SOC Interface (Zero Icons, Premium Unique Typography)
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            color: #f8fafc;
        }}

        .stApp {{
            background-color: #080c14;
        }}

        .soc-hero {{
            position: relative;
            background-image: 
                linear-gradient(180deg, rgba(8, 12, 20, 0.70) 0%, rgba(8, 12, 20, 0.88) 55%, rgba(8, 12, 20, 1.0) 100%),
                {bg_css};
            background-size: cover;
            background-position: center 35%;
            background-repeat: no-repeat;
            padding: 36px 28px 24px 28px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}

        .soc-hero-title {{
            font-family: 'Syne', sans-serif;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
            margin-bottom: 4px;
        }}

        .soc-hero-subtitle {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 13px;
            color: #94a3b8;
            max-width: 800px;
            line-height: 1.5;
        }}

        .soc-warning-box {{
            background-color: rgba(15, 23, 42, 0.8);
            border-left: 3px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 18px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .font-mono-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar Controls & Status ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div style='font-family: \"Syne\", sans-serif; font-size: 18px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;'>AI-NIDS // CONSOLE</div>", unsafe_allow_html=True)
        st.caption("NETWORK SECURITY OPERATIONS CENTER (SOC)")

        st.markdown("---")
        st.markdown("<div class='font-mono-tag' style='color: #94a3b8; margin-bottom: 8px;'>SYSTEM TELEMETRY</div>", unsafe_allow_html=True)
        predictor = get_unified_predictor()
        am = get_alert_manager()

        if predictor.binary_pipeline is not None:
            st.success("BINARY MODEL: ACTIVE [RECALL-PRIORITIZED]")
        else:
            st.error("BINARY MODEL: OFFLINE")

        if predictor.multiclass_pipeline is not None:
            st.success("MULTICLASS MODEL: ACTIVE [13 CLASSES]")
        else:
            st.error("MULTICLASS MODEL: OFFLINE")

        if predictor.anomaly_pipeline is not None:
            st.info("ANOMALY ENGINE: CALIBRATED [IFOREST]")
        else:
            st.warning("ANOMALY ENGINE: OFFLINE")

        st.caption("STORAGE: SQLITE LOCAL [data/alerts.db]")

        st.markdown("---")
        st.markdown("<div class='font-mono-tag' style='color: #94a3b8; margin-bottom: 6px;'>OPERATIONAL SPECIFICATION</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #64748b; line-height: 1.6;">
            &bull; ZERO CLOUD LATENCY<br>
            &bull; 100% OFFLINE LOCALHOST<br>
            &bull; PASSIVE NON-DESTRUCTIVE<br>
            &bull; ZERO PAYLOAD RETENTION
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Top Hero Banner with Background Image ─────────────────────────────────
    st.markdown(
        """
        <div class="soc-hero">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #3b82f6; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px;">
                PASSIVE MONITORING // ZERO-PAYLOAD PRIVACY
            </div>
            <div class="soc-hero-title">Security Operations Center & Alert Triage</div>
            <div class="soc-hero-subtitle">
                High-throughput flow classification, statistical outlier scoring, and Explainable AI (XAI) feature attributions executing offline on host infrastructure.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_compliance_notice()

    # ── Multi-Tab Dashboard Layout ────────────────────────────────────────────
    tabs = st.tabs([
        "[ TAB 1: OVERVIEW & TELEMETRY ]",
        "[ TAB 2: CSV INGESTION & INFERENCE ]",
        "[ TAB 3: PCAP TRAFFIC ANALYSIS ]",
        "[ TAB 4: EVALUATION & BENCHMARKS ]",
        "[ TAB 5: ALERT TRIAGE & REPORTING ]",
    ])

    # ==============================================================================
    # TAB 1: OVERVIEW & TELEMETRY
    # ==============================================================================
    with tabs[0]:
        st.markdown("<div class='font-mono-tag' style='color: #f8fafc; font-size: 13px; margin-bottom: 12px;'>OPERATIONAL TELEMETRY & SITUATIONAL AWARENESS</div>", unsafe_allow_html=True)

        stats = am.get_alert_statistics()

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Flows / Alerts", stats["total_alerts"])
        kpi2.metric("Unresolved Backlog", stats["unresolved_count"], delta=f"{stats['by_status'].get('New', 0)} New", delta_color="inverse")
        kpi3.metric("Confirmed Incidents", stats["confirmed_count"])
        kpi4.metric("False Positives", stats["false_positive_count"])
        kpi5.metric("Statistical Anomalies", stats["anomaly_count"])

        st.markdown("---")

        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            st.markdown("<div class='font-mono-tag' style='color: #94a3b8; margin-bottom: 6px;'>PREDICTED THREAT HYPOTHESES BY CLASS</div>", unsafe_allow_html=True)
            if stats["by_class"]:
                classes = list(stats["by_class"].keys())
                counts = list(stats["by_class"].values())

                fig, ax = plt.subplots(figsize=(6, 3.8))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')
                y_pos = np.arange(len(classes))
                ax.barh(y_pos, counts, color="#3b82f6", edgecolor="#2563eb")
                ax.set_yticks(y_pos)
                ax.set_yticklabels([c.upper() for c in classes], fontsize=9, color="#f8fafc", fontfamily="sans-serif")
                ax.tick_params(colors='#94a3b8')
                ax.invert_yaxis()
                ax.set_xlabel("Count", fontsize=9, color="#94a3b8")
                ax.spines['bottom'].set_color('#334155')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No alert records registered. Execute CSV or PCAP analysis to seed telemetry.")

        with col_chart2:
            st.markdown("<div class='font-mono-tag' style='color: #94a3b8; margin-bottom: 6px;'>ANALYST TRIAGE LIFECYCLE DISTRIBUTION</div>", unsafe_allow_html=True)
            if stats["by_status"]:
                statuses = list(stats["by_status"].keys())
                status_counts = list(stats["by_status"].values())

                fig, ax = plt.subplots(figsize=(5, 3.8))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')
                colors = ["#f59e0b", "#3b82f6", "#ef4444", "#10b981", "#64748b"]
                status_colors = [colors[i % len(colors)] for i in range(len(statuses))]

                wedges, texts, autotexts = ax.pie(
                    status_counts,
                    labels=[s.upper() for s in statuses],
                    autopct="%1.1f%%",
                    startangle=140,
                    colors=status_colors,
                    textprops=dict(color="#f8fafc", fontsize=9),
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No status distribution records available.")

        st.markdown("<div class='font-mono-tag' style='color: #94a3b8; margin-top: 16px; margin-bottom: 6px;'>RECENT ALERTS FEED (LAST 10 RECORDS)</div>", unsafe_allow_html=True)
        recent_alerts = am.get_alerts(limit=10)
        if not recent_alerts.empty:
            display_cols = [
                "id", "timestamp", "flow_id", "src_ip", "dst_ip", "predicted_class",
                "confidence_score", "is_anomaly", "status"
            ]
            available_cols = [c for c in display_cols if c in recent_alerts.columns]
            st.dataframe(recent_alerts[available_cols], use_container_width=True)
        else:
            st.caption("No alerts logged in database.")

    # ==============================================================================
    # TAB 2: CSV BATCH INFERENCE
    # ==============================================================================
    with tabs[1]:
        st.markdown("<div class='font-mono-tag' style='color: #f8fafc; font-size: 13px; margin-bottom: 8px;'>FLOW FEATURE CSV INGESTION & VALIDATION</div>", unsafe_allow_html=True)
        st.caption(
            "Ingests 66-column statistical flow feature sets (CIC-IDS2017 schema). "
            "AI-NIDS enforces strict schema validation, executes model inference, and computes XAI feature attributions."
        )

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Network Flow Feature CSV",
                type=["csv"],
                help="Requires 66 numeric flow features matching CIC-IDS2017 schema.",
            )
        with col_sample:
            st.write("&nbsp;")
            use_sample_csv = st.button("[ LOAD SAMPLE CSV ]", use_container_width=True)

        df_csv_input: Optional[pd.DataFrame] = None
        if uploaded_file is not None:
            try:
                df_csv_input = pd.read_csv(uploaded_file)
                st.success(f"CSV LOADED: {len(df_csv_input)} rows, {len(df_csv_input.columns)} columns.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        elif use_sample_csv:
            sample_path = Path("data/sample_features.csv")
            if sample_path.exists():
                df_csv_input = pd.read_csv(sample_path)
                st.info(f"Loaded bundled sample dataset: {len(df_csv_input)} rows, {len(df_csv_input.columns)} columns.")
            else:
                st.error("Sample dataset data/sample_features.csv not found.")

        if df_csv_input is not None:
            is_valid, missing, stripped = validate_csv_columns(df_csv_input)

            if stripped:
                st.info(f"Target/metadata columns separated to prevent leakage: {', '.join(stripped)}")

            if not is_valid:
                st.error(f"SCHEMA VALIDATION FAILED: Missing {len(missing)} required features:")
                st.code("\n".join(missing[:15]) + ("\n... and more" if len(missing) > 15 else ""))
                st.warning("Inference blocked to protect pipeline integrity. Please provide a compatible CICFlowMeter CSV.")
            else:
                st.success("SCHEMA VALIDATION PASSED: All 66 expected flow features verified.")

                col_opt1, col_opt2, col_opt3 = st.columns(3)
                with col_opt1:
                    model_type_choice = st.selectbox(
                        "Model Pipeline",
                        options=["multiclass", "binary"],
                        format_func=lambda x: "Multiclass (13 Attack Types)" if x == "multiclass" else "Binary (Benign/Malicious)",
                    )
                with col_opt2:
                    inc_anomaly = st.checkbox("Isolation Forest Anomaly Scoring", value=True)
                    inc_xai = st.checkbox("Generate XAI Feature Attributions", value=True)
                with col_opt3:
                    conf_slider = st.slider("Min Confidence Threshold", 0.0, 1.0, 0.50, 0.05)

                if st.button("[ EXECUTE THREAT ANALYSIS ]", type="primary"):
                    with st.spinner("Executing inference across model pipelines..."):
                        try:
                            features_only = df_csv_input.drop(columns=[c for c in stripped if c in df_csv_input.columns])
                            t0 = datetime.now()
                            res_df = predictor.predict(
                                features_only,
                                model_type=model_type_choice,
                                include_anomaly=inc_anomaly,
                                include_explanation=inc_xai,
                            )
                            duration = (datetime.now() - t0).total_seconds()
                            st.session_state["csv_predictions"] = res_df
                            st.session_state["csv_duration"] = duration
                            st.success(f"Inference complete: {len(res_df)} rows in {duration:.3f} s ({duration/max(1, len(res_df))*1000:.2f} ms/row).")
                        except Exception as exc:
                            st.error(f"Inference error: {exc}")

                if "csv_predictions" in st.session_state:
                    res_df = st.session_state["csv_predictions"]
                    lbl_col = "predicted_class" if "predicted_class" in res_df.columns else "label"

                    total_evaluated = len(res_df)
                    benign_count = sum(res_df[lbl_col].astype(str).str.lower() == "benign")
                    suspicious_count = total_evaluated - benign_count
                    anomaly_count = sum(res_df["is_anomaly"] == 1) if "is_anomaly" in res_df.columns else 0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Flows Evaluated", total_evaluated)
                    c2.metric("Benign Flows", benign_count)
                    c3.metric("Predicted Threats", suspicious_count)
                    c4.metric("Statistical Anomalies", anomaly_count)

                    show_cols = [c for c in [lbl_col, "score", "anomaly_score", "is_anomaly", "explanation"] if c in res_df.columns]
                    st.dataframe(res_df[show_cols].head(50), use_container_width=True)

                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("[ REGISTER FLAGGED THREATS TO DATABASE ]"):
                            alerts_added = am.create_alerts_from_df(
                                res_df,
                                model_version=f"{model_type_choice}_pipeline",
                                min_confidence=conf_slider,
                                filter_malicious_only=True,
                            )
                            st.success(f"Registered {alerts_added} alert(s) into SQLite repository.")

                    with col_act2:
                        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="[ DOWNLOAD PREDICTIONS CSV ]",
                            data=csv_bytes,
                            file_name=f"ai_nids_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )

    # ==============================================================================
    # TAB 3: PCAP TRAFFIC ANALYSIS
    # ==============================================================================
    with tabs[2]:
        st.markdown("<div class='font-mono-tag' style='color: #f8fafc; font-size: 13px; margin-bottom: 8px;'>NETWORK PACKET PCAP INGESTION & DISSECTION</div>", unsafe_allow_html=True)
        st.caption(
            "Ingests PCAP/PCAPNG network packet captures offline. Packets are parsed into "
            "bidirectional 5-tuple flows, mapped to 66 statistical features, and evaluated against trained models."
        )

        col_pcap_in, col_pcap_sample = st.columns([3, 1])
        with col_pcap_in:
            uploaded_pcap = st.file_uploader(
                "Upload PCAP / PCAPNG Trace File",
                type=["pcap", "pcapng", "cap"],
                help="Network packet traces. Processed strictly in-memory and offline.",
            )
        with col_pcap_sample:
            st.write("&nbsp;")
            use_sample_pcap = st.button("[ LOAD SAMPLE PCAP ]", use_container_width=True)

        target_pcap_path: Optional[Path] = None
        if uploaded_pcap is not None:
            temp_pcap = Path("/tmp") / f"uploaded_{uploaded_pcap.name}"
            with open(temp_pcap, "wb") as f:
                f.write(uploaded_pcap.getbuffer())
            target_pcap_path = temp_pcap
            st.success(f"PCAP file loaded: {uploaded_pcap.name} ({len(uploaded_pcap.getbuffer())} bytes).")
        elif use_sample_pcap:
            sample_pcap = PCAP_DIR / "sample_network_traffic.pcap"
            if sample_pcap.exists():
                target_pcap_path = sample_pcap
                st.info(f"Loaded bundled sample trace: {sample_pcap.name} ({sample_pcap.stat().st_size} bytes).")
            else:
                st.error(f"Sample PCAP not found at {sample_pcap}.")

        if target_pcap_path is not None:
            pcap_model = st.selectbox("Inference Model Pipeline", ["multiclass", "binary"], key="pcap_model_sel")
            if st.button("[ DISSECT PACKETS & ANALYZE FLOWS ]", type="primary"):
                with st.spinner("Reconstructing bidirectional flows and extracting 66 statistical features..."):
                    try:
                        df_flows, telemetry = process_pcap(
                            target_pcap_path,
                            predictor=predictor,
                            model_type=pcap_model,
                            include_anomaly=True,
                            include_explanation=True,
                        )
                        st.session_state["pcap_flows"] = df_flows
                        st.session_state["pcap_telemetry"] = telemetry
                        st.success("PCAP flow reconstruction and inference completed.")
                    except Exception as pcap_err:
                        st.error(f"PCAP processing failed: {pcap_err}")

        if "pcap_flows" in st.session_state and "pcap_telemetry" in st.session_state:
            df_flows = st.session_state["pcap_flows"]
            telem = st.session_state["pcap_telemetry"]

            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Packets Read", telem.get("packets_read", 0))
            t2.metric("Flows Formed", telem.get("flows_reconstructed", 0))
            t3.metric("Parse Time", f"{telem.get('parse_time_seconds', 0.0):.3f} s")
            t4.metric("Avg Latency / Flow", f"{telem.get('avg_prediction_latency_ms', 0.0):.2f} ms")
            t5.metric("Throughput", f"{telem.get('throughput_packets_per_sec', 0.0):.0f} pkts/s")

            if not df_flows.empty:
                view_cols = [
                    c for c in [
                        "flow_id", "src_ip", "src_port", "dst_ip", "dst_port", "protocol_name",
                        "packet_count", "byte_count", "duration_ms", "predicted_class", "label",
                        "score", "anomaly_score", "is_anomaly", "explanation"
                    ] if c in df_flows.columns
                ]
                st.dataframe(df_flows[view_cols], use_container_width=True)

                col_pcap_act1, col_pcap_act2 = st.columns(2)
                with col_pcap_act1:
                    if st.button("[ REGISTER PCAP THREATS TO DATABASE ]"):
                        alerts_added = am.create_alerts_from_df(
                            df_flows,
                            model_version=f"pcap_{pcap_model}_pipeline",
                            min_confidence=0.0,
                            filter_malicious_only=True,
                        )
                        st.success(f"Registered {alerts_added} flow alert(s) to SQLite repository.")

                with col_pcap_act2:
                    pcap_csv = df_flows.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="[ DOWNLOAD FLOW RECORDS CSV ]",
                        data=pcap_csv,
                        file_name=f"pcap_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("No IP flows could be reconstructed from the supplied PCAP file.")

    # ==============================================================================
    # TAB 4: EVALUATION & BENCHMARKS
    # ==============================================================================
    with tabs[3]:
        st.markdown("<div class='font-mono-tag' style='color: #f8fafc; font-size: 13px; margin-bottom: 8px;'>OFFLINE MODEL VALIDATION & EMPIRICAL BENCHMARKS</div>", unsafe_allow_html=True)
        st.caption("Empirical performance metrics, confusion matrices, and feature importances.")

        eval_subtabs = st.tabs([
            "[ BINARY CLASSIFIERS ]",
            "[ MULTICLASS ATTACK METRICS ]",
            "[ CONFUSION MATRICES ]",
            "[ FEATURE IMPORTANCE (XAI) ]",
            "[ LIMITATIONS & BIASES ]",
        ])

        with eval_subtabs[0]:
            cmp_file = REPORTS_DIR / "model_comparison.csv"
            if cmp_file.exists():
                df_cmp = pd.read_csv(cmp_file)
                st.dataframe(df_cmp, use_container_width=True)

                fig, ax = plt.subplots(figsize=(7, 3.5))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')
                models = df_cmp["model"].tolist()
                f1_scores = df_cmp["f1"].tolist()
                recalls = df_cmp["recall"].tolist()

                x = np.arange(len(models))
                width = 0.35
                ax.bar(x - width/2, f1_scores, width, label="F1 Score", color="#3b82f6")
                ax.bar(x + width/2, recalls, width, label="Recall", color="#10b981")
                ax.set_xticks(x)
                ax.set_xticklabels([m.upper() for m in models], color="#f8fafc")
                ax.tick_params(colors='#94a3b8')
                ax.set_ylabel("Score", color="#94a3b8")
                ax.set_ylim(0, 1.0)
                ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with eval_subtabs[1]:
            mc_file = REPORTS_DIR / "multiclass_per_class_metrics.csv"
            if mc_file.exists():
                df_mc = pd.read_csv(mc_file)
                mc_models = df_mc["model"].unique().tolist()
                sel_mc_model = st.selectbox("Select Model Architecture", mc_models)
                df_mc_sel = df_mc[df_mc["model"] == sel_mc_model]
                st.dataframe(df_mc_sel, use_container_width=True)

        with eval_subtabs[2]:
            cm_dir = REPORTS_DIR / "confusion_matrices"
            if cm_dir.exists():
                cm_images = sorted(list(cm_dir.glob("*.png")))
                if cm_images:
                    cm_options = {img.stem: img for img in cm_images}
                    sel_cm = st.selectbox("Select Matrix", list(cm_options.keys()))
                    st.image(str(cm_options[sel_cm]))

        with eval_subtabs[3]:
            fi_images = sorted(list(REPORTS_DIR.glob("feature_importance_*.png")))
            if fi_images:
                fi_options = {img.stem: img for img in fi_images}
                sel_fi = st.selectbox("Select Attribution Plot", list(fi_options.keys()))
                st.image(str(fi_options[sel_fi]))

        with eval_subtabs[4]:
            limitations_file = REPORTS_DIR / "model_limitations_and_biases.md"
            if limitations_file.exists():
                with open(limitations_file, "r", encoding="utf-8") as f:
                    content = f.read()
                st.markdown(content)

    # ==============================================================================
    # TAB 5: ALERT TRIAGE & REPORTING
    # ==============================================================================
    with tabs[4]:
        st.markdown("<div class='font-mono-tag' style='color: #f8fafc; font-size: 13px; margin-bottom: 8px;'>SECURITY ALERT MANAGEMENT & INCIDENT TRIAGE</div>", unsafe_allow_html=True)
        st.caption("Inspect threat hypotheses, update investigation statuses, record forensic findings, and export sanitized reports.")

        with st.expander("[ SEARCH & FILTER CONTROLS ]", expanded=True):
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            with fcol1:
                status_filter = st.selectbox("Filter Status", ["All"] + ALERT_STATUS_CHOICES)
                date_preset = st.selectbox("Date Window", ["All Time", "Today", "Last 7 Days", "Last 30 Days"])
            with fcol2:
                all_alerts_stats = am.get_alert_statistics()
                existing_classes = sorted(list(all_alerts_stats["by_class"].keys()))
                class_filter = st.selectbox("Filter Threat Class", ["All"] + existing_classes)
                anom_only = st.checkbox("Outliers Only", value=False)
            with fcol3:
                min_conf_filter = st.slider("Min Confidence Score", 0.0, 1.0, 0.0, 0.05)
                search_text = st.text_input("Keyword Search", placeholder="IP, Port, Flow ID, Notes...")
            with fcol4:
                records_limit = st.number_input("Max Query Records", min_value=10, max_value=5000, value=200, step=50)

        start_dt = None
        now_utc = datetime.now(timezone.utc)
        if date_preset == "Today":
            start_dt = now_utc.strftime("%Y-%m-%d")
        elif date_preset == "Last 7 Days":
            start_dt = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
        elif date_preset == "Last 30 Days":
            start_dt = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")

        filtered_alerts = am.get_alerts(
            start_date=start_dt,
            predicted_class=class_filter if class_filter != "All" else None,
            status=status_filter if status_filter != "All" else None,
            min_confidence=min_conf_filter if min_conf_filter > 0 else None,
            is_anomaly=True if anom_only else None,
            search_query=search_text if search_text else None,
            limit=records_limit,
        )

        st.markdown(f"<div class='font-mono-tag' style='color: #94a3b8; margin: 12px 0 6px 0;'>RETRIEVED {len(filtered_alerts)} ALERT RECORD(S)</div>", unsafe_allow_html=True)

        if not filtered_alerts.empty:
            table_cols = [
                "id", "timestamp", "flow_id", "src_ip", "src_port", "dst_ip", "dst_port",
                "protocol", "predicted_class", "confidence_score", "is_anomaly", "status", "analyst_notes"
            ]
            display_tbl = filtered_alerts[[c for c in table_cols if c in filtered_alerts.columns]]
            st.dataframe(display_tbl, use_container_width=True)

            st.markdown("---")
            st.markdown("<div class='font-mono-tag' style='color: #f8fafc; font-size: 13px; margin-bottom: 8px;'>INCIDENT INVESTIGATION & TRIAGE PANEL</div>", unsafe_allow_html=True)

            col_triage_sel, col_triage_act = st.columns([1, 2])

            with col_triage_sel:
                alert_ids = filtered_alerts["id"].tolist()
                selected_alert_id = st.selectbox("Select Alert Record", alert_ids)
                selected_row = filtered_alerts[filtered_alerts["id"] == selected_alert_id].iloc[0]

                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.08); font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                    <div>ALERT ID: #{selected_alert_id}</div>
                    <div>TIMESTAMP: {selected_row.get('timestamp')}</div>
                    <div>5-TUPLE: {selected_row.get('src_ip')}:{selected_row.get('src_port')} \u2192 {selected_row.get('dst_ip')}:{selected_row.get('dst_port')} ({selected_row.get('protocol')})</div>
                    <div>PREDICTED THREAT: {(selected_row.get('predicted_class') or '').upper()}</div>
                    <div>CONFIDENCE: {selected_row.get('confidence_score'):.2f}</div>
                    <div>OUTLIER: {'YES' if selected_row.get('is_anomaly') == 1 else 'NO'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_triage_act:
                current_status = selected_row.get("status", DEFAULT_ALERT_STATUS)
                status_index = ALERT_STATUS_CHOICES.index(current_status) if current_status in ALERT_STATUS_CHOICES else 0

                new_status = st.selectbox("Update Lifecycle Status", ALERT_STATUS_CHOICES, index=status_index, key="triage_status_sel")
                analyst_notes = st.text_area(
                    "Analyst Investigation Notes",
                    value=str(selected_row.get("analyst_notes", "")),
                    placeholder="Document firewall correlations, host validation, false positive reasoning...",
                    height=100,
                )

                expl_data = selected_row.get("explanation", "")
                if expl_data and expl_data != "{}":
                    with st.expander("[ XAI FEATURE ATTRIBUTION DETAILS ]"):
                        st.code(expl_data)

                if st.button("[ SAVE STATUS & ANALYST NOTES ]", type="primary"):
                    success = am.update_alert_status(
                        alert_id=selected_alert_id,
                        new_status=new_status,
                        notes=analyst_notes,
                    )
                    if success:
                        st.success(f"Alert #{selected_alert_id} updated to '{new_status}'.")
                        st.rerun()

            st.markdown("---")
            col_exp1, col_exp2 = st.columns([2, 1])
            with col_exp1:
                export_csv_bytes = filtered_alerts.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="[ DOWNLOAD SANITIZED ALERTS CSV REPORT ]",
                    data=export_csv_bytes,
                    file_name=f"ai_nids_alerts_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Exports non-payload flow telemetry and detection status for compliance auditing.",
                )
            with col_exp2:
                with st.expander("[ DATABASE RESET ]"):
                    if st.button("[ CLEAR ALL ALERTS ]"):
                        cleared_count = am.clear_all_alerts()
                        st.warning(f"Cleared {cleared_count} alerts from database.")
                        st.rerun()

        else:
            st.info("No alert records match current filter criteria.")


if __name__ == "__main__":
    main()
