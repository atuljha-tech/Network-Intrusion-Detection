# AI-NIDS: AI-Powered Network Intrusion Detection System

[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 175 Passed](https://img.shields.io/badge/Tests-175%20Passed-success.svg)](tests/)
[![Offline Local Execution](https://img.shields.io/badge/Cloud%20Dependencies-Zero%20(%E2%82%B90%20Cost)-brightgreen.svg)]()
[![Passive Traffic Analysis](https://img.shields.io/badge/Monitoring-Passive%20Non--Destructive-informational.svg)]()

> **Phases 1 through 7 Complete: Data Engineering, EDA, Feature Engineering, Supervised Binary & Multiclass Classification, Unsupervised Anomaly Detection, Explainable AI (XAI), PCAP Traffic Processing, SOC Analyst Dashboard, Generalization Auditing & Final Packaging.**

AI-NIDS is a zero-cost, local-only machine learning system designed to ingest raw network traffic (PCAP/PCAPNG) or extracted flow features (CIC-IDS2017 schema), reconstruct bidirectional network conversations, detect cyber threats, triage statistical anomalies, provide transparent feature attribution, and facilitate forensic investigation via a dedicated SOC dashboard.

---

## ⚠️ Operational Notice: Threat Hypotheses vs. Confirmed Incidents

* **Passive Non-Destructive Monitoring:** AI-NIDS operates strictly as a **passive monitor**. It never transmits packets, modifies routing tables, injects RST flags, or takes autonomous destructive actions.
* **Metadata-Only Privacy:** AI-NIDS computes statistical flow metrics (packet size distributions, inter-arrival times, flag counts, window sizes). It **never inspects, records, or stores application packet payloads or user credentials**.
* **Statistical Hypotheses:** Machine learning predictions are **probabilistic risk hypotheses**, not definitive proof of compromise. Every flagged detection defaults to status `New` and requires human security analyst triage and correlation.

---

## 📁 Repository Structure

```
NetworkIntrusionDetection/
├── dashboard/                     ← SOC Analyst Web Interfaces & REST API
│   ├── app.py                     ← Multi-tab interactive Streamlit dashboard
│   ├── server.py                  ← Lightweight Flask REST API server for SPA
│   └── static/                    ← Unified Single-Page Application (SPA) frontend
│       ├── index.html             ← High-density zero-icon SOC analyst console
│       ├── bg_image.jpg           ← Hero background texture & overlay
│       └── chart.umd.min.js       ← 100% offline local Chart.js library
├── run_dashboard.py               ← Single-command CLI launcher for the SPA console
├── bg_image.jpg                   ← Root visual asset for dashboard hero styling
├── data/
│   ├── raw/                       ← Place official CIC-IDS2017 CSV files here
│   ├── processed/                 ← Cleaned data, feature metadata JSON
│   ├── pcap/                      ← Network traces (.pcap / .pcapng)
│   │   └── sample_network_traffic.pcap ← Bundled RFC-compliant test capture
│   ├── sample_features.csv        ← Sample 66-feature dataset for testing
│   └── alerts.db                  ← Persistent local SQLite database (auto-created)
├── models/                        ← Fitted sklearn pipelines (.joblib) & metadata (.json)
│   ├── binary_intrusion_pipeline.joblib
│   ├── multiclass_intrusion_pipeline.joblib
│   ├── isolation_forest.joblib
│   ├── model_metadata.json
│   ├── multiclass_model_metadata.json
│   └── anomaly_metadata.json
├── notebooks/                     ← Jupyter research & exploratory notebooks
├── reports/                       ← Validation, benchmark, generalization & model reports
│   ├── classification_reports/    ← Per-model precision/recall/F1 text summaries
│   ├── confusion_matrices/        ← Binary & multiclass confusion matrix heatmaps
│   ├── feature_importance_*.png   ← Tree feature importance plots (MDI)
│   ├── generalization_evaluation_report.md ← Held-out cross-period evaluation
│   ├── performance_benchmark_report.md     ← Latency, throughput, memory audit
│   ├── performance_metrics.json   ← Machine-readable benchmarking telemetry
│   ├── model_limitations_and_biases.md     ← Rigorous bias, leakage & FP/FN analysis
│   ├── pcap_extractor_and_zeek_comparison.md ← Telemetry comparison against Zeek
│   ├── class_imbalance_report.txt ← Class frequency & Gini inequality analysis
│   └── anomaly_detection_report.txt ← Isolation Forest calibration report
├── src/
│   ├── __init__.py
│   ├── config.py                  ← Central configuration, paths, options, logging
│   ├── data_loader.py             ← CSV discovery, encoding fallback, schema validation
│   ├── preprocess.py              ← Data normalization, target derivation, deduplication
│   ├── validate_data.py           ← Pre-training data health diagnostics
│   ├── eda.py                     ← Exploratory data analysis & statistical distribution plots
│   ├── run_eda.py                 ← CLI entry point for EDA pipeline
│   ├── feature_engineering.py     ← Pipeline construction, leak-free preprocessing
│   ├── model_utils.py             ← Serialization, schema validation, package versioning
│   ├── train.py                   ← Supervised binary & multiclass training CLI
│   ├── evaluate.py                ← Independent class evaluation & generalization benchmarks
│   ├── multiclass.py              ← Multiclass model architectures & per-class metrics
│   ├── imbalance.py               ← Class imbalance quantification & Gini coefficients
│   ├── anomaly.py                 ← Unsupervised Isolation Forest outlier calibration
│   ├── explainability.py          ← Tree feature importance & local surrogate/SHAP attribution
│   ├── flow_features.py           ← Bi-directional 66-feature flow accumulator & extractor
│   ├── pcap_processor.py          ← PCAP streaming, flow reconstruction & inference engine
│   ├── alert_manager.py           ← SQLite persistence, triage lifecycle & sanitized reports
│   ├── predict.py                 ← Common unified prediction interface
│   └── benchmark.py               ← Performance, latency, throughput & memory profiler
├── tests/                         ← Automated Pytest Suite (175 tests across all modules)
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   ├── test_validate_data.py
│   ├── test_eda.py
│   ├── test_feature_engineering.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   ├── test_phase6.py
│   ├── test_phase7.py             ← Comprehensive integration, resilience & benchmark tests
│   └── test_dashboard_server.py   ← REST API & dashboard integration test suite
├── requirements.txt               ← Exact pinned dependency specifications
├── README.md                      ← System documentation & usage guide
└── .gitignore                     ← Excludes raw data, large captures, databases, caches
```

---

## 🛠️ Prerequisites & Local Environment Setup

AI-NIDS runs **100% locally on localhost** with zero paid dependencies, cloud APIs, or external database servers.

### 1. Prerequisites
* **Operating System:** macOS, Linux, or Windows (WSL recommended for Windows).
* **Python Version:** Python 3.10 or 3.11.
* **Memory:** Minimum 4 GB RAM (8 GB recommended for processing large PCAPs).

### 2. Installation
Clone the repository and set up a clean Python virtual environment:

```bash
# 1. Clone repository
git clone https://github.com/your-org/NetworkIntrusionDetection.git
cd NetworkIntrusionDetection

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

## 📦 Dataset Acquisition (CIC-IDS2017)

AI-NIDS uses the standard **CIC-IDS2017** network intrusion benchmark developed by the Canadian Institute for Cybersecurity (University of New Brunswick).

### How to Acquire the Real Dataset:
1. Download the generated CSVs from the official source:
   * **Official UNB Portal:** [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Extract the archive and place the CSV files into the `data/raw/` directory:
   ```
   data/raw/
   ├── Monday-WorkingHours.pcap_ISCX.csv
   ├── Tuesday-WorkingHours.pcap_ISCX.csv
   ├── Wednesday-workingHours.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```

### Synthetic Data Fallback (Zero-Data Quickstart):
If raw CSVs are not yet downloaded, AI-NIDS **automatically generates a synthetic multi-day dataset** containing all 66 CICFlowMeter statistical features and attack distributions. Add the `--synthetic` flag to any CLI command to force synthetic mode.

---

## 🚀 Reproducible Execution Commands

### 1. Run Automated Test Suite (175 Tests)
AI-NIDS features comprehensive test coverage covering data loading, preprocessing, model serialization, traffic processing, alert storage, schema compatibility, server REST API endpoints, and generalization:

```bash
python3 -m pytest tests/ -v
```

### 2. Run Exploratory Data Analysis & Feature Engineering
Generates distribution plots, missing value heatmaps, correlation matrices, and fits feature pipelines:

```bash
python3 -m src.run_eda --synthetic
```

### 3. Train Machine Learning Models
Trains binary classifiers (Logistic Regression, Decision Tree, Random Forest), multiclass classifiers, and unsupervised Isolation Forest anomaly detection:

```bash
# Train all pipelines (Binary, Multiclass, and Isolation Forest):
python3 -m src.train --all --synthetic

# Or train individual paradigms:
python3 -m src.train --synthetic             # Binary classification only
python3 -m src.train --multiclass --synthetic # Multiclass classification only
python3 -m src.train --anomaly --synthetic    # Isolation Forest anomaly experiment only
```

### 4. Run Generalization & Held-Out Capture Audit
Evaluates cross-period domain shift and novel attack detection on held-out temporal scenarios:

```bash
python3 -c "from src.evaluate import evaluate_held_out_generalization; evaluate_held_out_generalization(force_synthetic=True)"
```

### 5. Run Performance & Latency Benchmarks
Measures single-flow inference latency (p50, p95, p99), batch throughput, PCAP dissection rate, and memory usage:

```bash
python3 -m src.benchmark
```

### 6. Launch the SOC Analyst Dashboard

AI-NIDS offers two complementary interfaces:

#### Option A: High-Density Single-Page Application (SPA) Console (Recommended)
A modern, zero-icon, typography-driven cybersecurity workstation powered by a Flask REST backend with 100% offline local Chart.js visualizations:

```bash
# Launch on default port 8501:
python3 run_dashboard.py

# Or specify a custom port:
python3 run_dashboard.py 8080
```
Open your web browser at: `http://127.0.0.1:8501`

#### Option B: Streamlit Multi-Tab Workspace
```bash
streamlit run dashboard/app.py
```
Open your web browser at: `http://localhost:8501`

---

## 🖥️ SOC Analyst Console Capabilities & Workflow

The Unified Single-Page Application (`run_dashboard.py` / `dashboard/server.py`) provides 6 high-density visual zones designed specifically for security operations centers:

1. **Top Navigation Bar:**
   * System health status indicators (Database connection, ML pipeline health).
   * Active operational profile (`PASSIVE MONITORING // METADATA-ONLY`).
   * Offline indicator and quick-refresh telemetry trigger.
2. **Executive Telemetry & KPI Cards:**
   * Total Alerts Logged, Unresolved Backlog (`New` / `Investigating`), Confirmed Incidents, Statistical Outliers (Isolation Forest), and Median Inference Latency.
3. **Ingestion & Threat Analysis Panel:**
   * Dual-mode drag-and-drop ingestion supporting both 66-feature CSV datasets and raw `.pcap` / `.pcapng` network traces.
   * One-click preloaded sample buttons: `[ LOAD SAMPLE CSV ]` and `[ LOAD SAMPLE PCAP ]`.
   * Inference toggles: Model Architecture (`Multiclass 13-Class` vs `Binary`), `Isolation Forest Anomaly Scoring`, and `Tree-Surrogate Explainability (XAI)`.
   * Strict schema validation enforcing 66 CIC-IDS2017 features, stripping target/leakage columns (`label`, `is_malicious`), and returning granular validation feedback.
4. **Forensic Telemetry Charts (100% Offline):**
   * **Threat Hypothesis Distribution:** Doughnut chart breaking down detected attack categories.
   * **Analyst Triage Lifecycle:** Bar chart illustrating alert state progression (`New`, `Investigating`, `Confirmed Incident`, `False Positive`, `Resolved`).
   * **Empirical Performance Benchmarks:** Latency comparisons (Binary vs Multiclass vs Anomaly vs XAI) and batch flow throughput.
5. **Security Alerts & Triage Workspace:**
   * Real-time filterable data grid with keyword search across 5-tuple IPs, ports, and protocols.
   * Multi-select filters for lifecycle status and threat category.
   * **Forensic Inspection Modal:** Inspect full 5-tuples, anomaly scores, calibrated probabilities, and XAI feature attributions.
   * **In-Place Lifecycle State Machine:** Transition alert status (`New` ➔ `Investigating` ➔ `Confirmed Incident` / `False Positive` / `Resolved`) with mandatory audit notes saved directly to `data/alerts.db`.
   * **Sanitized CSV Compliance Export:** Download filtered forensic reports without exposing packet payloads.
6. **Governance & System Compliance Footer:**
   * Operational boundary reminders, zero cloud dependency verification, and ₹0 budget certification.

---

## ⏱️ Measured Empirical Performance Benchmarks

All performance metrics below were measured locally on the host machine (`Darwin x86_64, Python 3.11.9`) without mock timers:

| Metric Category | Component / Operation | Measured Value |
|---|---|---|
| **Single-Flow Latency** | Binary Classifier (`LogisticRegression`/`RandomForest`) | **10.73 ms** (median) / 23.89 ms (p95) |
| **Single-Flow Latency** | Multiclass Classifier (`DecisionTree`/`RandomForest`) | **10.98 ms** (median) / 33.01 ms (p95) |
| **Single-Flow Latency** | Unsupervised Isolation Forest Anomaly Scoring | **30.18 ms** (median) / 67.68 ms (p95) |
| **Single-Flow Latency** | Tree-Surrogate Feature Attribution (XAI) | **14.12 ms** (median) / 60.74 ms (p95) |
| **Batch Inference** | Batch size = 100 flows | **6,636.5 flows/second** |
| **Batch Inference** | Batch size = 1,000 flows | **21,973.1 flows/second** |
| **Batch Inference** | Batch size = 5,000 flows | **38,320.5 flows/second** |
| **Traffic Throughput** | PCAP Packet Dissection & Flow Tracking | **1,727.6 packets/second** |
| **Traffic Throughput** | End-to-End PCAP Processing (Dissect + ML + Anomaly) | **378.3 packets/second** (155.8 flows/s) |
| **Memory Footprint** | Process Resident Set Size (RSS) | **218.58 MB** |
| **Memory Footprint** | Peak Tracemalloc Heap Allocation (5,000 flows) | **18.82 MB** |

*Complete benchmark logs and scaling curves are preserved in [`reports/performance_benchmark_report.md`](reports/performance_benchmark_report.md) and [`reports/performance_metrics.json`](reports/performance_metrics.json).*

---

## 🔬 Implemented Capabilities vs. Planned / Future Work

To maintain absolute scientific and engineering integrity, system capabilities are explicitly categorized below:

### ✅ Implemented & Verified Capabilities:
1. **66-Feature Bidirectional Flow Accumulator:** Computes exact CIC-IDS2017 statistical features from PCAP/PCAPNG packet streams (`src/flow_features.py`).
2. **Leak-Free ML Pipelines:** Encapsulated imputation and scaling fitted strictly on training data; target columns (`label`, `is_malicious`) and metadata stripped before inference (`src/model_utils.py`).
3. **Supervised Binary & Multiclass Classification:** Recall-prioritized binary models and 13-category attack models with class weighting (`src/train.py`, `src/multiclass.py`).
4. **Calibrated Unsupervised Anomaly Scoring:** Isolation Forest trained exclusively on nominal benign baselines (`src/anomaly.py`).
5. **Explainable AI (XAI):** Tree feature importance (MDI) and local surrogate attribution explaining top contributing flow factors (`src/explainability.py`).
6. **SOC Web Dashboard:** Multi-tab Streamlit analyst workspace (`dashboard/app.py`).
7. **Local SQLite Alert Repository:** Persistent database with analyst lifecycle transitions (`New` ➔ `Investigating` ➔ `Confirmed Incident` / `False Positive` / `Resolved`) and SQL injection immunity (`src/alert_manager.py`).
8. **Sanitized CSV Compliance Exports:** Exports flow telemetry while guaranteeing zero payload or credential exposure.
9. **Cross-Period Generalization Audit:** Automated evaluation across temporal capture periods (`reports/generalization_evaluation_report.md`).
10. **Zero Paid Services:** 100% offline local execution using open-source Python libraries.

### 🔮 Planned / Future Enhancements (Not Implemented):
1. **Active Inline Intrusion Prevention (IPS):** AI-NIDS is strictly passive; active TCP RST teardown or automated firewall route injection is not implemented.
2. **Deep Packet Inspection (DPI) & TLS Decryption:** AI-NIDS inspects Layer 3/4 statistical flow metadata only; it does not decrypt HTTPS or inspect TLS payload content.
3. **Hardware Acceleration:** Kernel-bypass (e.g. DPDK, PF_RING) or FPGA acceleration for 10Gbps+ wire-speed streaming is not implemented (traffic parsing relies on Python/Scapy streaming).
4. **Distributed Inference:** Multi-node Spark/Ray clustering is not implemented; designed for single-node SOC workstation deployments.

---

## 🛡️ Security, Privacy & Ethical Compliance

* **No Commits of Network Captures:** Only a synthetic, RFC-compliant test PCAP (`sample_network_traffic.pcap`) with private documentation IPs is committed. Raw network traces are ignored by `.gitignore`.
* **Zero Payload Retention:** Packet payloads are discarded immediately after computing statistical header lengths.
* **No Hardcoded Secrets or Cloud Keys:** The project requires no API keys, cloud tokens, or remote authentication.
* **Analyst in the Loop:** AI-NIDS operates as a decision-support aid. Human verification is strictly required before treating a threat hypothesis as a confirmed incident.

---

## 📝 License

This project is licensed under the MIT License. It is intended for defensive cybersecurity research, education, and authorized network monitoring.
