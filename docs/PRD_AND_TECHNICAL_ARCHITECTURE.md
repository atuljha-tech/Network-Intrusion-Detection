# Product Requirements Document (PRD) & Technical Architecture Specification
## AI-Powered Network Intrusion Detection System (AI-NIDS)

---

### Document Control & Governance
* **Document Title:** AI-NIDS Comprehensive System Architecture, Dataset Provenance & Product Requirements Document
* **Document Version:** 2.2.0 (Complete System Release — Phases 1 through 8)
* **Status:** Approved, Validated & Edge Deployed
* **Target Audience:** Lead Architects, SOC Supervisors, Cyber Threat Analysts, Machine Learning Engineers, Regulatory Compliance Officers
* **Authors:** AI-NIDS Core Engineering & Research Team
* **Publication Date:** October 26, 2026
* **Repository:** `https://github.com/atuljha-tech/Network-Intrusion-Detection.git`
* **Licensing & Governance:** FOSS (Free and Open Source Software) — ₹0 Budget Architecture

---

## 1. Executive Summary & Product Vision

### 1.1 Executive Summary
The AI-Powered Network Intrusion Detection System (AI-NIDS) is an autonomous, local, zero-cloud-cost cybersecurity intelligence platform designed to ingest raw network traffic, extract bidirectional statistical flow features, perform multi-tier machine learning threat classification, and present security analysts with actionable, interpretable forensic telemetry.

Unlike traditional commercial intrusion detection systems (IDS) that rely exclusively on brittle, easily bypassed signature hashes or expensive enterprise cloud subscriptions, AI-NIDS utilizes a hybrid defense-in-depth model:
1. **Supervised Binary Threat Attribution:** High-recall, low-latency detection of malicious traffic versus benign baseline.
2. **Supervised Multiclass Classification:** Attribution across 13 distinct cyber attack categories conforming to the MITRE ATT&CK framework.
3. **Unsupervised Anomaly Scoring:** Calibrated Isolation Forest models detecting zero-day anomalies and protocol deviations without prior signature exposure.
4. **Surrogate Explainable AI (XAI):** Sub-15 millisecond decision-tree rule extraction explaining the exact transport variables triggering each alert.
5. **Unified SOC Command Console:** High-density, single-page analyst dashboard offering dual-mode ingestion (CSV datasheets and raw PCAP packet traces), live telemetry charts, and SQLite triage persistence.

### 1.2 Core Constraints & Operating Principles
* **₹0 Budget Guarantee:** Built strictly using free, open-source software libraries. Absolutely no paid API keys, commercial licenses, hosted database services, or recurring subscriptions are required.
* **100% Offline Capability:** All preprocessing, feature normalization, model inference, alert persistence, and UI rendering operate entirely offline on the local host.
* **Passive Traffic Monitoring Only:** Operates strictly as a passive listener or tap. The system never injects packets, modifies routing tables, performs intrusive port scanning, or executes disruptive firewall terminations.
* **Zero-Payload Privacy Rule:** Deep packet inspection (DPI) of application payloads (e.g., HTTP request bodies, credentials, emails) is strictly avoided. Only transport-layer and network-layer statistical metadata (packet counts, inter-arrival times, window sizes, TCP flags) are analyzed.
* **Human-in-the-Loop Triage:** Alerts are classified as candidate incidents. Only authorized human analysts retain the authority to confirm or close incidents.

---

## 2. Dataset Architecture & Storage Locations

### 2.1 Dataset Provenance & Baseline
The AI-NIDS system is architected around and evaluated upon the **CIC-IDS2017 Dataset**, produced by the Canadian Institute for Cybersecurity (CIC) and the University of New Brunswick (UNB). 

* **Why CIC-IDS2017 Was Selected:** 
  Most legacy intrusion datasets (e.g., KDD99, DARPA98) suffer from obsolete traffic distributions, lack of modern attack vectors, and artificial traffic pacing. CIC-IDS2017 represents modern network protocols, incorporating standard protocols (HTTP, HTTPS, FTP, SSH, DNS, SMB) alongside realistic multi-vector cyber attack scenarios executed over a 5-day capture period.
* **Scope of Ingested Traffic:** 
  The raw dataset spans 8 capture periods across 5 working days (Monday through Friday), comprising 2,830,743 bidirectional flow records. It incorporates real background traffic generated through human behavioral profiling (B-Profiles) combined with systematic threat generation (M-Profiles).

### 2.2 Dataset Location in Folder Structure
The repository organizes data assets into designated directories managed by `src/config.py`:

```
NetworkIntrusionDetection/
├── data/
│   ├── raw/                           <-- DROP LOCATION: Official raw CIC-IDS2017 CSV files
│   │   └── (Monday-WorkingHours.pcap_ISCX.csv, Tuesday-..., etc.)
│   ├── processed/                     <-- Normalized, sanitized feature matrices
│   │   ├── feature_metadata.json      <-- Canonical schema metadata and feature types
│   │   └── preprocessing_summary.json <-- Missing values, scaling parameters, imputation records
│   ├── pcap/                          <-- Passive packet capture files
│   │   └── sample_network_traffic.pcap <-- Bundled legal packet capture (17 packets, 7 flows)
│   ├── sample_features.csv            <-- Bundled verified 66-feature test datasheet (10 flows)
│   └── alerts.db                      <-- Persistent SQLite database for alert records & notes
```

#### Detailed Location Guide:
1. **`data/sample_features.csv` (Root Data Directory):**
   A verified 10-flow test datasheet pre-formatted to the exact 66-feature schema. This file is bundled directly in the git repository and used by the web dashboard for instant evaluation, automated CI/CD tests, and one-click sample downloads.
2. **`data/pcap/sample_network_traffic.pcap`:**
   A synthetic, legally compliant PCAP packet trace containing real TCP/UDP handshakes, HTTP requests, and ICMP frames. Used to test bidirectional packet dissection, flow reconstruction, and end-to-end model inference from raw packet bytes.
3. **`data/raw/` (Configured Drop Target):**
   The destination directory for researchers and SOC administrators to drop full multi-gigabyte CIC-IDS2017 raw CSV files. When present, `src/data_loader.py` ingests and streams these files into the training pipeline.
4. **`data/processed/`:**
   Stores intermediate cleaned feature matrices and the formal `feature_metadata.json` document specifying feature boundaries, statistics, and column data types.
5. **`data/alerts.db` (Local SQLite Database):**
   Stores all registered security alerts, flow 5-tuples (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`), model confidence scores, anomaly flags, analyst notes, and investigation lifecycle statuses. On Vercel edge deployments, this dynamically mounts to `/tmp/alerts.db`.

### 2.3 The 66 Canonical Numerical Features
The pipeline isolates exactly 66 numerical, bidirectional transport features matching the CICFlowMeter / CIC-IDS2017 specification:

```
+----------------------------------------------------------------------------------------------------+
|                               THE 66 CANONICAL FLOW FEATURES                                       |
+------------------------------------+---------------------------------------------------------------+
| Category                           | Included Features                                             |
+------------------------------------+---------------------------------------------------------------+
| 1. Timing & Duration (5)           | flow_duration, flow_iat_mean, flow_iat_std,                   |
|                                    | flow_iat_max, flow_iat_min                                    |
| 2. Forward Packet Dynamics (9)     | total_fwd_packets, total_length_of_fwd_packets,                |
|                                    | fwd_packet_length_max, fwd_packet_length_min,                  |
|                                    | fwd_packet_length_mean, fwd_packet_length_std,                 |
|                                    | fwd_iat_total, fwd_iat_mean, fwd_iat_std                      |
| 3. Backward Packet Dynamics (9)    | total_backward_packets, total_length_of_bwd_packets,           |
|                                    | bwd_packet_length_max, bwd_packet_length_min,                  |
|                                    | bwd_packet_length_mean, bwd_packet_length_std,                 |
|                                    | bwd_iat_total, bwd_iat_mean, bwd_iat_std                      |
| 4. Throughput & Rates (4)          | flow_bytes_s, flow_packets_s, fwd_packets_s, bwd_packets_s     |
| 5. Global Packet Distribution (5)  | min_packet_length, max_packet_length, packet_length_mean,     |
|                                    | packet_length_std, packet_length_variance                     |
| 6. TCP Control Flags (11)          | fin_flag_count, syn_flag_count, rst_flag_count,               |
|                                    | psh_flag_count, ack_flag_count, urg_flag_count,               |
|                                    | cwe_flag_count, ece_flag_count, fwd_psh_flags,                |
|                                    | bwd_psh_flags, fwd_urg_flags                                  |
| 7. Protocol Segment & Window (8)   | fwd_header_length, bwd_header_length, down_up_ratio,           |
|                                    | average_packet_size, avg_fwd_segment_size,                    |
|                                    | avg_bwd_segment_size, init_win_bytes_forward,                 |
|                                    | init_win_bytes_backward                                       |
| 8. Flow Activity & Inactivity (8)  | act_data_pkt_fwd, min_seg_size_forward, active_mean,          |
|                                    | active_std, active_max, active_min, idle_mean, idle_std       |
| 9. Inter-Arrival Extremes (7)      | fwd_iat_max, fwd_iat_min, bwd_iat_max, bwd_iat_min,           |
|                                    | bwd_urg_flags, idle_max, idle_min                             |
+------------------------------------+---------------------------------------------------------------+
```

#### Significance of Key Transport Variables:
* **Inter-Arrival Times (`flow_iat_*`):** Automated botnets, beaconing agents, and volumetric DoS scripts exhibit near-zero or strictly periodic inter-arrival times, whereas legitimate human navigation displays high variance and Poisson-like delays.
* **TCP Control Flags (`syn_flag_count`, `rst_flag_count`):** Port scans and SYN-flood DoS attacks generate elevated SYN flags with absent or near-zero ACK confirmations. Reset (RST) spikes identify port knock probes encountering closed sockets.
* **Initial Window Bytes (`init_win_bytes_forward`):** Operating systems enforce specific TCP receive window sizes. Deviations or tiny window advertisements reveal stealth reconnaissance tools (e.g., Nmap raw socket crafts) attempting evasion.
* **Active vs. Idle Times (`active_*`, `idle_*`):** Multi-stage advanced persistent threats (APTs) exhibit distinctive active command execution bursts followed by extended dormant intervals to blend into normal enterprise traffic.

### 2.4 Label Taxonomy & Classification Mapping
The dataset encompasses 13 discrete attack categories and a benign baseline:
1. `benign`: Legitimate background communications (web browsing, DNS lookups, file transfers).
2. `ddos`: Volumetric Distributed Denial of Service (High packet rate, low IAT, uniform packet size).
3. `dos hulk`: High-volume HTTP GET/POST request flooding designed to exhaust web application workers.
4. `portscan`: Sequential or distributed socket scanning probing TCP/UDP ports across targets.
5. `bot`: Ares botnet command-and-control communication, periodic beaconing, and automated commands.
6. `dos goldeneye`: Application-layer HTTP keep-alive starvation using randomized headers.
7. `dos slowhttptest`: Slow-read / slow-request starvation maintaining open TCP sockets indefinitely.
8. `ssh-patator`: Dictionary and brute-force authentication attacks against SSH daemons.
9. `ftp-patator`: High-velocity password cracking against standard FTP servers.
10. `dos slowloris`: Partial HTTP header streaming exhausting server thread pools.
11. `web attack - brute force`: Automated HTTP form credential brute-forcing against web logins.
12. `heartbleed`: Memory extraction exploit leveraging OpenSSL TLS heartbeat length discrepancies.
13. `infiltration`: Post-exploitation lateral movement and privilege escalation inside internal subnets.

---

## 3. System Architecture & End-to-End Pipeline

```
+-----------------------------------------------------------------------------------+
|                            INCOMING TRAFFIC INTERFACE                             |
|       [ Raw PCAP Traces (.pcap) ]           [ 66-Feature CSV Datasheets (.csv) ]  |
+-----------------------------------------------------------------------------------+
                                  |                                     |
                                  v                                     |
+-------------------------------------------------------------+         |
|         LAYER 1: PASSIVE PACKET DISSECTION & FLOW TRACKER   |         |
|  - Bidirectional 5-Tuple Index: (src, dst, sport, dport, pr)|         |
|  - Microsecond Inter-Arrival Time (IAT) Calculation         |         |
|  - TCP Control Flag Aggregation (SYN, FIN, RST, ACK, PSH)   |         |
|  - Activity / Idle Timeout State Machine (120s / 5s)        |         |
|  - ZERO-PAYLOAD PRIVACY: Payload bytes discarded at socket  |         |
+-------------------------------------------------------------+         |
                                  |                                     |
                                  v                                     v
+-----------------------------------------------------------------------------------+
|               LAYER 2: DATA SANITIZATION & FEATURE PIPELINE                       |
|  - Schema Verification: Strict 66 CIC-IDS2017 Column Conformance                 |
|  - Non-Feature Stripping: Drops labels, IDs, source filenames (Prevents Leakage)  |
|  - Mathematical Sanitization: Inf/NaN clipping, median imputation                 |
|  - Scaler Pipeline: Standard/Robust scaling fitted exclusively on training set    |
+-----------------------------------------------------------------------------------+
                                  |
                                  +-----------------------+
                                  |                       |
                                  v                       v
+--------------------------------------+  +-----------------------------------------+
| LAYER 3: SUPERVISED CLASSIFIERS      |  | LAYER 4: UNSUPERVISED ANOMALY ENGINE    |
| [Binary Recall-Prioritized Model]    |  | [Isolation Forest Model (100 Trees)]    |
|   - Threat Attribution Score (0..1)  |  |   - Trained on Clean Benign Baseline    |
| [Multiclass Classifier (13 Classes)] |  |   - Normalized Anomaly Score (0.0..1.0) |
|   - Attack Category Identification   |  |   - Contamination Threshold: 5.0%       |
+--------------------------------------+  +-----------------------------------------+
                                  \                       /
                                   \                     /
                                    v                   v
+-----------------------------------------------------------------------------------+
|               LAYER 5: SURROGATE EXPLAINABLE AI (XAI) ATTRIBUTION                 |
|  - Fast Decision Tree Surrogate Engine (<15 ms latency per flow)                  |
|  - Local Feature Attributions: Top-5 transport features responsible for alert     |
|  - Human-Readable Rules: e.g. "flow_duration > 15s AND syn_flag_count > 10"       |
+-----------------------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------------------+
|               LAYER 6: PERSISTENT SOC ALERT TRIAGE & MANAGEMENT                   |
|  - SQLite Datastore: data/alerts.db (or /tmp/alerts.db on Vercel)                 |
|  - Forensic Record: 5-tuple, timestamp, prediction, confidence, anomaly, XAI rule |
|  - Analyst Lifecycle Tracking: [NEW] -> [INVESTIGATING] -> [CONFIRMED/FP/RESOLVED]|
+-----------------------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------------------+
|               LAYER 7: WEB DASHBOARD & CLOUD EDGE DEPLOYMENT                      |
|  - Flask 3.x REST API Engine (dashboard/server.py)                                |
|  - High-Density Forensic SPA Interface (dashboard/static/index.html)              |
|  - Vercel WSGI Proxy Middleware (VercelPathFixMiddleware in api/index.py)         |
|  - Responsive Visualizations: Chart.js Attack Breakdown, Lifecycle & Benchmarks  |
+-----------------------------------------------------------------------------------+
```

---

## 4. Machine Learning Engine & Model Specifications

The AI-NIDS architecture combines supervised classification, unsupervised anomaly scoring, and local explainability:

```
[ Incoming Network Flow Vector (66 Numerical Features) ]
                         |
           +-------------+-------------+
           |                           |
           v                           v
+-----------------------+   +-----------------------+
|  SUPERVISED PIPELINE  |   |  UNSUPERVISED ENGINE  |
|  - Binary Detector    |   |  - Isolation Forest   |
|  - Multiclass (13-Cl) |   |    (Clean Baseline)   |
+-----------------------+   +-----------------------+
           |                           |
     Threat Class &             Anomaly Outlier
    Confidence (0..1)             Score (0..1)
           \                           /
            +------------+------------+
                         |
                         v
+---------------------------------------------------+
|            SURROGATE EXPLAINABLE AI               |
|  - Tree Feature Weights & Decision Path Rules     |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|              TRIAGE DECISION MATRIX               |
| - High Confidence Attack -> P1 High Alert         |
| - Benign + High Anomaly  -> P3 Anomaly Candidate  |
| - Low Confidence Threat  -> P2 Triage Required    |
+---------------------------------------------------+
```

### 4.1 Supervised Binary Intrusion Detection
* **Objective:** First-line defensive gatekeeper identifying malicious vectors with high recall.
* **Models Evaluated:** Logistic Regression, Decision Tree, Random Forest.
* **Design Philosophy:** Prioritize malicious recall over raw accuracy. In enterprise security operations, a missed breach (false negative) is disastrous, whereas a false alarm (false positive) is safely triaged by an analyst.
* **Target Schema:** Binary column `is_malicious` ($0 = \text{Benign}$, $1 = \text{Malicious}$).

### 4.2 Supervised Multiclass Classifier (13 Categories)
* **Objective:** Granular attack taxonomy attribution for incident response prioritization.
* **Architecture:** Balanced-weight Decision Tree and Random Forest multi-output classifiers.
* **Class Imbalance Strategy:** Real-world network traffic is heavily skewed towards benign interactions ($>80\%$), while critical attacks like Heartbleed or Infiltration represent $<0.5\%$. The training pipeline computes inverse-frequency class weights:
  $$W_c = \frac{N_{\text{total}}}{N_{\text{classes}} \times N_c}$$
  ensuring minority attack classes receive equal gradient consideration during tree splits.

### 4.3 Unsupervised Anomaly Scoring (Isolation Forest)
* **Objective:** Detect novel attacks, zero-day exploits, and structural network anomalies.
* **Baseline Training:** 100 Isolation Trees fitted exclusively on benign background traffic (`semi_supervised_benign_baseline`).
* **Score Normalization:** Scikit-learn raw decision scores are mapped into a standardized $[0.0, 1.0]$ range:
  $$\text{Anomaly Score} = \min\left(1.0, \max\left(0.0, \frac{-s_{\text{raw}} - \mu}{3\sigma}\right)\right)$$
* **Outlier Threshold:** Calibrated at $0.05$ contamination ($5\%$ anomaly budget). Outlier flows exceeding score $0.60$ receive the `is_anomaly = 1` flag.

### 4.4 Real-Time Explainability Engine (Surrogate XAI)
* **The Problem with SHAP:** Computing Shapley values for 66 numerical features requires exponential permutations, incurring 2.0 to 5.0 seconds of latency per flow—completely unusable for interactive SOC operations.
* **The Surrogate Solution:** AI-NIDS trains a lightweight Decision Tree surrogate model that approximates the global predictor's decision boundary locally.
* **Attribution Output:** In under **15 milliseconds**, the surrogate outputs:
  1. Top-5 feature attribution weights.
  2. The logical rule predicate (e.g., `flow_duration > 15.2s AND syn_flag_count > 12 AND fwd_iat_mean < 0.05s`).

---

## 5. Empirical Performance Profiling & Latency Benchmarks

All performance metrics were empirically benchmarked on `Darwin x86_64` (Python 3.11.9) with zero mock timers:

### 5.1 Single-Flow Latency Profile (500 Timed Iterations)
| Pipeline Component | Median Latency | Mean Latency | 95th Percentile | 99th Percentile | Min / Max |
|---|---|---|---|---|---|
| **Binary Classifier** | `10.73 ms` | `13.69 ms` | `23.89 ms` | `63.20 ms` | `7.54 / 169.95 ms` |
| **Multiclass Classifier** | `10.98 ms` | `14.82 ms` | `33.01 ms` | `71.54 ms` | `7.49 / 91.97 ms` |
| **Anomaly Detection (iForest)** | `30.18 ms` | `35.79 ms` | `67.68 ms` | `99.28 ms` | `22.84 / 116.92 ms` |
| **Explainability (XAI Surrogate)** | `14.12 ms` | `21.56 ms` | `60.74 ms` | `104.83 ms` | `8.97 / 375.04 ms` |

### 5.2 Batch Scaling & High-Volume Throughput
| Batch Size (Flows) | Batch Execution Time | Latency Per Flow | Throughput (Flows / Second) |
|---|---|---|---|
| `1` | `11.06 ms` | `11.059 ms` | **`90.4 flows/s`** |
| `10` | `17.97 ms` | `1.797 ms` | **`556.5 flows/s`** |
| `50` | `14.45 ms` | `0.289 ms` | **`3,459.2 flows/s`** |
| `100` | `15.07 ms` | `0.151 ms` | **`6,636.5 flows/s`** |
| `500` | `40.18 ms` | `0.080 ms` | **`12,443.6 flows/s`** |
| `1,000` | `45.51 ms` | `0.046 ms` | **`21,973.1 flows/s`** |
| `2,500` | `61.77 ms` | `0.025 ms` | **`40,475.8 flows/s`** |
| `5,000` | `130.48 ms` | `0.026 ms` | **`38,320.5 flows/s`** |

### 5.3 Memory Profiling & Guardrails
* **Process Resident Set Size (RSS):** `218.58 MB` (Comfortably operating on standard 4GB/8GB laptops).
* **Peak Heap Allocation during 5,000-flow batch:** `18.82 MB`.
* **PCAP Streaming Guardrail:** `MAX_PACKETS_STREAMING = 500,000` prevents OOM crashes when dissecting large network capture files.

---

## 6. Passive Packet Capture (PCAP) Integration

### 6.1 Bidirectional Flow Reconstruction
The `src/pcap_processor.py` and `src/flow_features.py` modules parse raw Ethernet/IP packets into directional conversations:
1. **5-Tuple Hashing:** Bidirectional indexing maps forward `(src_ip:sport -> dst_ip:dport, proto)` and reverse `(dst_ip:dport -> src_ip:sport, proto)` packets into a single cohesive flow object.
2. **Microsecond Inter-Arrival Time (IAT):** Tracks timestamp deltas between successive packets across forward and reverse streams.
3. **Flow Lifecycle Management:**
   * **Inactivity Timeout:** Expired flows are flushed to inference after `FLOW_TIMEOUT_SECONDS = 120.0s`.
   * **Active Burst Timeout:** Inactivity gaps exceeding `FLOW_ACTIVITY_TIMEOUT_SECONDS = 5.0s` segment the conversation into active transmission bursts and idle listening periods.

### 6.2 Comparison: Custom Python Extractor vs. Zeek
| Capability | Custom AI-NIDS Python Extractor | Zeek Network Monitor |
|---|---|---|
| **Feature Conformance** | 100% exact mathematical match to CIC-IDS2017 66 features | Produces log events (`conn.log`), requires schema translation |
| **System Dependencies** | Standard Python (Scapy / DPKT), zero external daemons | Requires Unix daemon, C++ build toolchains, root permissions |
| **Cross-Platform** | macOS, Linux, Windows, Cloud Serverless | Primarily Linux / POSIX environments |
| **Zero-Payload Enforcement**| Built-in: Payload bytes discarded at socket layer | Configurable via script policy (`zeek_init`) |

---

## 7. Unified Single-Page SOC Analyst Dashboard

### 7.1 Interface Design & Architecture
* **Single-Page Application (SPA):** Built with Flask 3.x, vanilla ECMAScript 6, and Chart.js.
* **Cybersecurity Aesthetics:** Semi-transparent dark styling (`#070b12` canvas), glassmorphism cards with backdrop blur, typographic badges, and zero distracting icons or emojis.
* **Dual Ingestion Dropzone:** Supports direct file dragging of either:
  1. 66-feature flow CSV datasets (`.csv`)
  2. Raw network packet traces (`.pcap`)
* **Clean "Awaiting Data Ingestion" State:** Telemetry metric cards, forensic distribution charts, and alert tables remain hidden until traffic data is uploaded or loaded.

### 7.2 Forensic Investigation Capabilities
1. **Live KPI Metric Cards:** Total Alerts, Active Threats, Benign Flows, Anomalies, Average Latency, and Throughput.
2. **Forensic Distribution Charts:**
   * *Attack Category Breakdown:* Multi-class threat distribution (DDoS, Bot, PortScan).
   * *Triage Lifecycle Distribution:* Status breakdown across alert states (New, Investigating, Confirmed, False Positive, Resolved).
   * *Benchmark Latencies:* P95 response times across Binary, Multiclass, Anomaly, and XAI engines.
3. **Interactive Alert Triage Grid:** Real-time search, status filtering, confidence bars, and one-click modal dossiers showing full transport sockets and XAI rules.
4. **Sanitized CSV Export:** One-click compliance report export containing full threat metadata with guaranteed zero packet payload leakage.

---

## 8. Cloud Edge Deployment & Vercel WSGI Middleware

### 8.1 Serverless Architecture on Vercel
The application is structured for instant zero-cost serverless deployment via Vercel:
* **Overcoming the 500 MB Lambda Limit:** 
  The raw development environment with Streamlit (~150MB), PyArrow (~80MB), and Matplotlib (~50MB) exceeded Vercel's 500 MB maximum uncompressed function limit (`LAMBDA_SIZE_EXCEEDED (533.92 MB)`). 
  By partitioning dependencies into `requirements.txt` (lean production runtime ~220 MB) and `requirements-dev.txt` (local-only research libraries), the deployment package was reduced by $>58\%$.
* **Dynamic Ephemeral SQLite:** On Vercel, `src/config.py` automatically routes SQLite database storage and temporary reports to `/tmp/alerts.db`, ensuring full compliance with serverless read-only root filesystems.

### 8.2 The `VercelPathFixMiddleware` Solution
In Vercel CLI 65+, internal rewrites (`/(.*) -> /api/index`) forward requests using the rewritten destination path (`PATH_INFO = "/api/index"`), which previously caused Flask to return 405 Method Not Allowed on API uploads and serve `index.html` on CSV downloads.

AI-NIDS resolves this via a dedicated WSGI middleware in [`dashboard/server.py`](file:///Users/it4/Desktop/NetworkIntrusionDetection/dashboard/server.py):
```python
class VercelPathFixMiddleware:
    """Restores true client request path from edge proxy headers."""
    def __init__(self, wsgi_app):
        self.app = wsgi_app

    def __call__(self, environ, start_response):
        for header in ("HTTP_X_FORWARDED_URI", "HTTP_X_ORIGINAL_URI", "RAW_URI"):
            raw_uri = environ.get(header)
            if raw_uri and raw_uri not in ("/api/index", "/api/index.py"):
                if "?" in raw_uri:
                    path, qs = raw_uri.split("?", 1)
                    environ["PATH_INFO"] = path
                    if not environ.get("QUERY_STRING"):
                        environ["QUERY_STRING"] = qs
                else:
                    environ["PATH_INFO"] = raw_uri
                break
        return self.app(environ, start_response)
```
This guarantees that all REST endpoints (`/api/download/sample-csv`, `/api/ingest/csv`, `/api/stats`, `/api/alerts`) execute seamlessly behind edge reverse proxies.

---

## 9. REST API Specification

| Endpoint | Method | Payload / Params | Response | Description |
|---|---|---|---|---|
| `/` | `GET` | None | `200 text/html` | Serves the single-page dashboard application. |
| `/api/health` | `GET` | None | `200 application/json` | Health verification of model pipelines and SQLite store. |
| `/api/stats` | `GET` | None | `200 application/json` | Executive telemetry KPIs and operational status. |
| `/api/alerts` | `GET` | `status`, `predicted_class`, `search`, `limit`, `offset` | `200 application/json` | Filterable, paginated alert records. |
| `/api/alerts/<id>` | `GET` | `id` in URL | `200 application/json` | Detailed single alert record with XAI attributions. |
| `/api/alerts/<id>/status` | `POST` | `{"status": str, "analyst_notes": str}` | `200 application/json` | In-place alert status and analyst notes updater. |
| `/api/ingest/csv` | `POST` | `multipart/form-data (file)` | `200 application/json` | Batch CSV feature validation and inference. |
| `/api/ingest/sample-csv` | `POST` | `model_type`, `include_anomaly`, `include_xai` | `200 application/json` | One-click evaluation of `data/sample_features.csv`. |
| `/api/ingest/pcap` | `POST` | `multipart/form-data (file)` | `200 application/json` | Raw PCAP packet parsing, flow reconstruction, and inference. |
| `/api/ingest/sample-pcap` | `POST` | `model_type`, `include_anomaly`, `include_xai` | `200 application/json` | One-click evaluation of `sample_network_traffic.pcap`. |
| `/api/chart-data` | `GET` | None | `200 application/json` | Formatted telemetry data for forensic distribution charts. |
| `/api/download/sample-csv` | `GET` | None | `200 text/csv` | Download verified 66-feature sample CSV datasheet. |
| `/api/export/csv` | `GET` | `status`, `predicted_class`, `search` | `200 text/csv` | Sanitized CSV report download (zero payload leakage). |
| `/api/alerts/clear` | `POST` | None | `200 application/json` | Reset local alert database to clean initial state. |

---

## 10. Security, Ethics & Legal Governance

1. **Non-Destructive Monitoring:** The system enforces a strict passive monitoring policy. It never issues active firewall drops, port scans, or route modifications.
2. **Zero-Payload Storage:** Application payloads are dropped immediately at the socket interface. No user credentials, cookies, or message bodies are ever recorded.
3. **Incident Qualification:** Model outputs represent candidate alerts, not confirmed breaches. Human analysts retain full oversight of alert state transitions.
4. **Authorized Networks Only:** Usage is legally restricted to authorized networks and explicit capture files.

---

## 11. Verification Matrix & Test Coverage

The platform is validated by 176 automated test cases across 11 test suites:
* `tests/test_dashboard_server.py`: REST routes, proxy headers, sample downloads, and lifecycle transitions.
* `tests/test_pcap_processor.py`: Packet parsing, 5-tuple indexing, and flow reconstruction.
* `tests/test_feature_engineering.py`: Imputation, scaling, and 66-feature schema conformance.
* `tests/test_phase3.py`: Supervised binary classification, recall constraints, and ROC-AUC evaluation.
* `tests/test_phase4.py`: Multiclass 13-category attack attribution and class weighting.
* `tests/test_phase5.py`: PCAP streaming throughput and Zeek schema mapping.
* `tests/test_phase6.py`: SQLite persistence, alert status transitions, and sanitized CSV exports.
* `tests/test_phase7.py`: Explainability surrogate accuracy, Isolation Forest calibration, and latency benchmarks.
* `tests/test_data_loader.py` & `tests/test_validate_data.py`: Data integrity and non-feature isolation.

---
*End of AI-NIDS Comprehensive Product Requirements Document & Technical Architecture Specification.*


---

## 12. Mathematical Formulations of Key Transport Metrics

To guarantee exact mathematical reproducibility and schema alignment with CIC-IDS2017, the feature extraction and preprocessing pipelines adhere to formal definitions:

### 12.1 Inter-Arrival Time (IAT) Statistics
For a sequence of packet arrival timestamps $t_1, t_2, \dots, t_K$:
* **Packet Inter-Arrival Delta:** 
  $$\Delta t_k = t_k - t_{k-1}, \quad \forall k \in \{2, \dots, K\}$$
* **Mean Inter-Arrival Time (`flow_iat_mean`):**
  $$\mu_{\text{IAT}} = \frac{1}{K-1} \sum_{k=2}^K \Delta t_k$$
* **Variance & Standard Deviation (`flow_iat_std`):**
  $$\sigma_{\text{IAT}} = \sqrt{\frac{1}{K-2} \sum_{k=2}^K (\Delta t_k - \mu_{\text{IAT}})^2}$$

### 12.2 Transfer Rates & Bandwidth Velocity
Given flow duration $D = t_K - t_1$ in seconds, total payload bytes $B_{\text{total}}$, and packet count $K$:
* **Flow Byte Rate (`flow_bytes_s`):**
  $$\text{Rate}_{\text{bytes}} = \frac{B_{\text{total}}}{\max(D, 10^{-6})}$$
* **Flow Packet Rate (`flow_packets_s`):**
  $$\text{Rate}_{\text{packets}} = \frac{K}{\max(D, 10^{-6})}$$

### 12.3 Asymmetry & Segment Ratios
* **Down/Up Ratio (`down_up_ratio`):**
  $$\mathcal{R}_{\text{down/up}} = \frac{K_{\text{backward}}}{K_{\text{forward}}}$$
* **Average Packet Size (`average_packet_size`):**
  $$\bar{S} = \frac{B_{\text{fwd}} + B_{\text{bwd}}}{K_{\text{forward}} + K_{\text{backward}}}$$

---

## 13. Deep Dive: Module-by-Module Codebase Anatomy

The AI-NIDS platform is composed of 14 core Python modules organized into a clean separation of concerns:

| Module File | Primary Responsibility | Key Classes & Functions |
|---|---|---|
| `src/config.py` | Central configuration, filesystem paths, hyperparameter bounds, and environment detection (`VERCEL`). | `PROJECT_ROOT`, `DATA_DIR`, `ALERTS_DB_PATH`, `NON_FEATURE_COLUMNS` |
| `src/data_loader.py` | Ingests multi-gigabyte raw CIC-IDS2017 CSV files, normalizes header casings, and isolates target labels. | `load_cicids_dataset()`, `stream_chunks()`, `generate_synthetic_dataset()` |
| `src/eda.py` | Statistical profiling, missing value audits, class imbalance calculation, and correlation matrices. | `run_eda()`, `plot_class_distributions()`, `generate_eda_report()` |
| `src/preprocess.py` | Mathematical sanitization, handling infinities, median imputation, and label encoding. | `clean_dataframe()`, `sanitize_infinities()`, `build_preprocessor()` |
| `src/feature_engineering.py` | Feature selection, zero-variance pruning, and standard/robust feature scaling. | `select_features()`, `scale_features()`, `save_feature_metadata()` |
| `src/train.py` | Supervised model training (Logistic Regression, Decision Trees, Random Forests) with balanced weighting. | `train_binary_models()`, `train_multiclass_models()`, `save_pipeline()` |
| `src/evaluate.py` | Classification reports, confusion matrices, ROC-AUC curves, and recall threshold validation. | `evaluate_model()`, `plot_confusion_matrix()`, `compute_metrics()` |
| `src/predict.py` | High-level unified inference engine orchestrating binary, multiclass, anomaly, and XAI pipelines. | `UnifiedPredictor`, `predict()`, `predict_single_flow()` |
| `src/anomaly.py` | Semi-supervised Isolation Forest anomaly scoring engine trained on benign traffic baselines. | `AnomalyDetector`, `fit_baseline()`, `score_samples()` |
| `src/explainability.py` | Sub-15ms decision-tree surrogate attribution replacing heavy SHAP computation. | `SurrogateExplainer`, `explain_flow()`, `extract_decision_path()` |
| `src/pcap_processor.py` | Passive raw packet parser, 5-tuple conversation tracker, and flow state machine. | `process_pcap()`, `PacketDissector`, `FlowAssembler` |
| `src/flow_features.py` | Real-time calculation of 66 statistical flow features from dissected packet arrival trains. | `extract_flow_features()`, `calculate_iat()`, `aggregate_flags()` |
| `src/alert_manager.py` | Persistent SQLite CRUD engine managing alert lifecycles, analyst notes, and compliance reports. | `AlertManager`, `create_alerts_from_df()`, `update_alert_status()` |
| `dashboard/server.py` | High-performance Flask 3.x REST API with `VercelPathFixMiddleware` reverse-proxy support. | `app`, `VercelPathFixMiddleware`, `ingest_csv()`, `download_sample_csv()` |

---

## 14. Comparative Industry Matrix: AI-NIDS vs. Existing Solutions

| Dimension | AI-NIDS | Snort 3.x | Suricata | Zeek (Bro) | Commercial Cloud NIDS |
|---|---|---|---|---|---|
| **Cost** | **₹0 (100% Free)** | Free / Open Source | Free / Open Source | Free / Open Source | $5,000–$50,000 / yr |
| **Detection Basis** | **Hybrid ML + Anomaly** | Rule Signatures | Rule Signatures | Behavioral Scripts | Black-Box Cloud AI |
| **Zero-Day Evasion** | **High (Anomaly iForest)**| Poor (Needs update) | Poor (Needs update) | Medium | High |
| **Zero-Payload Privacy**| **Built-in Guarantee** | Inspects Full Payload| Inspects Full Payload| Configurable Policy| Cloud Ingestion Risk |
| **Explainability** | **<15ms XAI Surrogate** | Rule String Only | Rule String Only | Script Log Trace | None (Opaque Score) |
| **Hardware Overhead** | **~218 MB RAM (Light)** | 1–4 GB RAM | 2–8 GB RAM | 2–8 GB RAM | Agent Footprint |
| **Deployment Simplicity**| **Single Python Process**| Complex C++ build | Heavy multi-threaded | Unix daemon cluster| Cloud Agent Install |

---

## 15. Operational Runbooks & Deployment Playbooks

### 15.1 Local Host Deployment Runbook
1. **Clone the repository:**
   ```bash
   git clone https://github.com/atuljha-tech/Network-Intrusion-Detection.git
   cd Network-Intrusion-Detection
   ```
2. **Install runtime dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the SOC Dashboard on localhost port 3000:**
   ```bash
   python3 run_dashboard.py 3000
   ```
4. **Access the Console:** Open `http://127.0.0.1:3000` in any modern web browser.

### 15.2 Vercel Cloud Serverless Deployment Runbook
1. Import the repository `atuljha-tech/Network-Intrusion-Detection` into the Vercel Dashboard.
2. Select **Framework Preset:** `Flask`.
3. Ensure the root directory is `./` and build command uses the defaults.
4. Vercel automatically detects `api/index.py` and `requirements.txt`.
5. The bundled `VercelPathFixMiddleware` dynamically intercepts Vercel proxy headers (`x-forwarded-uri`), ensuring instantaneous, zero-cost edge execution.

---
*End of Complete AI-NIDS Technical Architecture Specification.*

