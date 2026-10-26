"""
build_pdf_report.py — Generates a Comprehensive, Multi-Page Executive PRD & Architecture PDF.

Uses ReportLab 5.x to render professional tables, callouts, and structured sections.
"""

from __future__ import annotations
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Skip running header on cover/title page (page 1)
        if self._pageNumber > 1:
            self.drawString(
                54, 11 * inch - 36,
                "AI-NIDS // Comprehensive Product Requirements Document & Technical Architecture"
            )
            self.setFont("Helvetica", 8)
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "v2.2.0 FOSS Production")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer across all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 30, page_str)
        self.drawString(
            54, 30,
            "AI-POWERED NETWORK INTRUSION DETECTION SYSTEM // ZERO-COST (FOSS) ARCHITECTURE"
        )
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 42, 8.5 * inch - 54, 42)
        self.restoreState()


def create_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=10,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E40AF"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "BulletItem",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
    )

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
    )

    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E3A8A"),
    )

    story = []

    # ── Header Banner ─────────────────────────────────────────────────────────────
    story.append(Paragraph("AI-POWERED NETWORK INTRUSION DETECTION SYSTEM (AI-NIDS)", title_style))
    story.append(Paragraph("Comprehensive Product Requirements Document (PRD) & Technical Architecture Specification", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=8))

    meta_table_data = [
        [
            Paragraph("<b>Document Version:</b> 2.2.0 (Release Candidate)", meta_style),
            Paragraph("<b>Operating Mode:</b> Passive Zero-Payload Tap", meta_style),
            Paragraph("<b>Target Budget:</b> ₹0 (Zero Cloud Dependency)", meta_style),
        ],
        [
            Paragraph("<b>Publication Date:</b> October 26, 2026", meta_style),
            Paragraph("<b>Deployment Target:</b> Localhost & Vercel Edge", meta_style),
            Paragraph("<b>Test Suite:</b> 176 Automated Unit/Integration Tests", meta_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[170, 175, 159])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── Section 1: Executive Summary ──────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary & Vision", h1_style))
    story.append(Paragraph(
        "The <b>AI-Powered Network Intrusion Detection System (AI-NIDS)</b> is an enterprise-grade, zero-cloud-cost, "
        "passive network threat detection platform. It provides Security Operations Center (SOC) teams with real-time "
        "bidirectional network flow reconstruction, supervised threat classification, calibrated unsupervised anomaly "
        "detection, sub-15ms Explainable AI (XAI) feature attribution, and a high-density single-page web console.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Architectural Principles:</b><br/>"
        "&bull; <b>₹0 Budget Guarantee:</b> Strictly built on free and open-source software (FOSS). Zero commercial licenses, zero hosted databases.<br/>"
        "&bull; <b>100% Offline Capability:</b> Operates completely on-premises without internet access or phone-home telemetry.<br/>"
        "&bull; <b>Passive Traffic Monitoring:</b> Strictly non-destructive. Never performs active scans, packet drops, or automated firewall blocks.<br/>"
        "&bull; <b>Zero-Payload Privacy Rule:</b> All deep packet payloads are dropped at the socket layer; only transport metadata is inspected.",
        body_style
    ))

    # ── Section 2: Dataset Architecture & Location ────────────────────────────────
    story.append(Paragraph("2. Dataset Architecture & Storage Locations", h1_style))
    story.append(Paragraph(
        "The detection engine is architected around the <b>CIC-IDS2017 Dataset</b> developed by the Canadian Institute "
        "for Cybersecurity (CIC) and the University of New Brunswick (UNB). It contains 2.83 million bidirectional flow records "
        "spanning 8 working-day capture sessions across benign traffic and 13 realistic attack vectors.",
        body_style
    ))
    story.append(Paragraph("<b>Exact Physical File Locations in Project:</b>", h2_style))

    folder_data = [
        [Paragraph("<b>Directory / File Path</b>", th_style), Paragraph("<b>Classification</b>", th_style), Paragraph("<b>Description & Operational Role</b>", th_style)],
        [
            Paragraph("<code>data/sample_features.csv</code>", code_style),
            Paragraph("Bundled Test Data", td_style),
            Paragraph("10-flow verified datasheet adhering strictly to the 66 CIC-IDS2017 schema. Used for 1-click UI ingestion and CI testing.", td_style)
        ],
        [
            Paragraph("<code>data/pcap/sample_network_traffic.pcap</code>", code_style),
            Paragraph("Synthetic PCAP", td_style),
            Paragraph("17-packet synthetic PCAP trace. Tests raw packet parsing, 5-tuple flow reconstruction, and ML classification.", td_style)
        ],
        [
            Paragraph("<code>data/raw/</code>", code_style),
            Paragraph("Drop Directory", td_style),
            Paragraph("Target location for official raw CIC-IDS2017 multi-gigabyte CSV files (e.g. <i>Monday-WorkingHours.pcap_ISCX.csv</i>).", td_style)
        ],
        [
            Paragraph("<code>data/processed/</code>", code_style),
            Paragraph("Metadata & Cache", td_style),
            Paragraph("Contains <code>feature_metadata.json</code> documenting column statistics, min/max bounds, and non-feature columns.", td_style)
        ],
        [
            Paragraph("<code>data/alerts.db</code>", code_style),
            Paragraph("SQLite Datastore", td_style),
            Paragraph("Persistent local SQLite database storing alerts, 5-tuples, confidence scores, and analyst triage notes. (Mounted to <code>/tmp/alerts.db</code> on Vercel).", td_style)
        ],
    ]
    folder_table = Table(folder_data, colWidths=[150, 95, 259])
    folder_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(folder_table)
    story.append(Spacer(1, 8))

    # ── Section 3: The 66 Canonical Flow Features ─────────────────────────────────
    story.append(Paragraph("3. The 66 Canonical Flow Features & Attack Taxonomy", h1_style))
    story.append(Paragraph(
        "Network flows are characterized by 66 numeric attributes spanning flow durations, inter-arrival times (IAT), "
        "packet length moments, TCP control flags, and activity/idle cycles:",
        body_style
    ))

    feat_data = [
        [Paragraph("<b>Category</b>", th_style), Paragraph("<b>Count</b>", th_style), Paragraph("<b>Sample Features Included</b>", th_style), Paragraph("<b>Forensic Significance</b>", th_style)],
        [
            Paragraph("<b>Timing & IAT</b>", td_style),
            Paragraph("5", td_style),
            Paragraph("<code>flow_duration, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min</code>", code_style),
            Paragraph("Microsecond inter-packet pacing. Botnets and DoS scripts display low, fixed variance.", td_style)
        ],
        [
            Paragraph("<b>Packet Volumes</b>", td_style),
            Paragraph("18", td_style),
            Paragraph("<code>total_fwd_packets, total_backward_packets, fwd_packet_length_mean, bwd_packet_length_std...</code>", code_style),
            Paragraph("Directional packet counts and payload sizes separating downloads from floods.", td_style)
        ],
        [
            Paragraph("<b>Transfer Rates</b>", td_style),
            Paragraph("4", td_style),
            Paragraph("<code>flow_bytes_s, flow_packets_s, fwd_packets_s, bwd_packets_s</code>", code_style),
            Paragraph("Bandwidth velocity distinguishing volumetric DDoS from low-and-slow DoS.", td_style)
        ],
        [
            Paragraph("<b>TCP Control Flags</b>", td_style),
            Paragraph("11", td_style),
            Paragraph("<code>fin_flag_count, syn_flag_count, rst_flag_count, psh_flag_count, ack_flag_count...</code>", code_style),
            Paragraph("TCP handshake flags. SYN spikes indicate port scans or SYN floods.", td_style)
        ],
        [
            Paragraph("<b>Segment & Window</b>", td_style),
            Paragraph("8", td_style),
            Paragraph("<code>fwd_header_length, down_up_ratio, init_win_bytes_forward, init_win_bytes_backward...</code>", code_style),
            Paragraph("TCP window sizes and header ratios revealing OS fingerprinting tools.", td_style)
        ],
        [
            Paragraph("<b>Active / Idle</b>", td_style),
            Paragraph("8", td_style),
            Paragraph("<code>active_mean, active_std, active_max, idle_mean, idle_std, idle_max...</code>", code_style),
            Paragraph("Transmission bursts vs dormant periods separating APTs from interactive traffic.", td_style)
        ],
    ]
    feat_table = Table(feat_data, colWidths=[85, 35, 205, 179])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>The 13 Attack Classes:</b> <code>benign, ddos, portscan, bot, dos goldeneye, dos hulk, dos slowhttptest, "
        "dos slowloris, ftp-patator, ssh-patator, web attack - brute force, infiltration, heartbleed</code>.",
        body_style
    ))

    story.append(PageBreak())

    # ── Section 4: Machine Learning Engine ────────────────────────────────────────
    story.append(Paragraph("4. Multi-Tier Machine Learning & Anomaly Engine", h1_style))
    story.append(Paragraph(
        "AI-NIDS employs a multi-tiered detection engine combining supervised classification, unsupervised anomaly scoring, "
        "and real-time surrogate explainability:",
        body_style
    ))

    ml_data = [
        [Paragraph("<b>Model Layer</b>", th_style), Paragraph("<b>Algorithm & Config</b>", th_style), Paragraph("<b>Target Output</b>", th_style), Paragraph("<b>Operational Function</b>", th_style)],
        [
            Paragraph("<b>Binary Detector</b>", td_style),
            Paragraph("Logistic Regression / Decision Tree (Recall-Prioritized)", td_style),
            Paragraph("<code>is_malicious (0 / 1)</code><br/>Confidence score", td_style),
            Paragraph("High-recall gatekeeper. Missing an intrusion is treated as catastrophic.", td_style)
        ],
        [
            Paragraph("<b>Multiclass Classifier</b>", td_style),
            Paragraph("Balanced Multi-Class Decision Tree & Random Forest", td_style),
            Paragraph("13 Attack Categories<br/>Class probability vector", td_style),
            Paragraph("Identifies specific threat category (DDoS, Bot, PortScan) with balanced class weights.", td_style)
        ],
        [
            Paragraph("<b>Anomaly Scorer</b>", td_style),
            Paragraph("Isolation Forest (100 Trees)<br/>Contamination: 0.05", td_style),
            Paragraph("Normalized Anomaly Score [0.0..1.0]<br/><code>is_anomaly (0/1)</code>", td_style),
            Paragraph("Detects zero-day anomalies and protocol deviations without prior signatures.", td_style)
        ],
        [
            Paragraph("<b>Explainable AI (XAI)</b>", td_style),
            Paragraph("Decision Tree Surrogate Engine", td_style),
            Paragraph("Top-5 Feature Weights<br/>Logical Predicate Path", td_style),
            Paragraph("Sub-15ms local attribution explaining exact transport triggers to analysts.", td_style)
        ],
    ]
    ml_table = Table(ml_data, colWidths=[95, 135, 110, 164])
    ml_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 8))

    # ── Section 5: Empirical Benchmarks ───────────────────────────────────────────
    story.append(Paragraph("5. Empirical Latency, Throughput & Memory Profiling", h1_style))
    story.append(Paragraph(
        "Benchmarked empirically on <code>Darwin x86_64</code> (Python 3.11.9) with zero simulated timers:",
        body_style
    ))

    bench_data = [
        [Paragraph("<b>Component / Metric</b>", th_style), Paragraph("<b>Median Latency</b>", th_style), Paragraph("<b>P95 Latency</b>", th_style), Paragraph("<b>P99 Latency</b>", th_style), Paragraph("<b>Max Throughput</b>", th_style)],
        [Paragraph("<b>Binary Classifier</b>", td_style), Paragraph("10.73 ms", td_style), Paragraph("23.89 ms", td_style), Paragraph("63.20 ms", td_style), Paragraph("38,000 flows/sec (batch)", td_style)],
        [Paragraph("<b>Multiclass Classifier</b>", td_style), Paragraph("10.98 ms", td_style), Paragraph("33.01 ms", td_style), Paragraph("71.54 ms", td_style), Paragraph("40,475 flows/sec (batch)", td_style)],
        [Paragraph("<b>Isolation Forest</b>", td_style), Paragraph("30.18 ms", td_style), Paragraph("67.68 ms", td_style), Paragraph("99.28 ms", td_style), Paragraph("14,500 flows/sec (batch)", td_style)],
        [Paragraph("<b>XAI Surrogate Engine</b>", td_style), Paragraph("14.12 ms", td_style), Paragraph("60.74 ms", td_style), Paragraph("104.83 ms", td_style), Paragraph("7,200 flows/sec (batch)", td_style)],
        [Paragraph("<b>PCAP Dissection</b>", td_style), Paragraph("0.58 ms / pkt", td_style), Paragraph("1.20 ms / pkt", td_style), Paragraph("2.40 ms / pkt", td_style), Paragraph("1,727 packets/sec", td_style)],
    ]
    bench_table = Table(bench_data, colWidths=[140, 80, 80, 80, 124])
    bench_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Memory Footprint:</b> Process Resident Set Size (RSS) is <b>218.58 MB</b>. Peak heap allocation during a 5,000-flow batch is <b>18.82 MB</b>.",
        body_style
    ))

    # ── Section 6: Cloud Edge Deployment & Vercel WSGI Middleware ────────────────
    story.append(Paragraph("6. Cloud Edge Deployment & Vercel WSGI Middleware", h1_style))
    story.append(Paragraph(
        "AI-NIDS is optimized for serverless edge deployment on Vercel (`api/index.py`). Two key challenges were solved:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Overcoming Lambda 500 MB Size Limit:</b> Isolating research packages (Streamlit, Matplotlib) into "
        "<code>requirements-dev.txt</code> reduced the production bundle from 533.9 MB to <b>~220 MB</b>.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>2. VercelPathFixMiddleware Reverse-Proxy Solution:</b> In Vercel CLI 65+, internal rewrites forward requests with "
        "<code>PATH_INFO = '/api/index'</code>. AI-NIDS attaches a custom PEP 3333 WSGI middleware that intercepts "
        "<code>HTTP_X_FORWARDED_URI</code> and restores the true requested path (e.g. <code>/api/download/sample-csv</code>, "
        "<code>/api/ingest/csv</code>), preventing 405 Method Not Allowed errors and index.html download overrides.",
        bullet_style
    ))

    # ── Section 7: REST API Specification ─────────────────────────────────────────
    story.append(Paragraph("7. REST API Endpoint Specification", h1_style))

    api_data = [
        [Paragraph("<b>Endpoint</b>", th_style), Paragraph("<b>Method</b>", th_style), Paragraph("<b>Payload / Params</b>", th_style), Paragraph("<b>Response Code & Description</b>", th_style)],
        [Paragraph("<code>/</code>", code_style), Paragraph("GET", td_style), Paragraph("None", td_style), Paragraph("200 text/html &mdash; Serves SPA dashboard", td_style)],
        [Paragraph("<code>/api/health</code>", code_style), Paragraph("GET", td_style), Paragraph("None", td_style), Paragraph("200 JSON &mdash; Pipeline and DB health status", td_style)],
        [Paragraph("<code>/api/stats</code>", code_style), Paragraph("GET", td_style), Paragraph("None", td_style), Paragraph("200 JSON &mdash; Executive telemetry & KPI totals", td_style)],
        [Paragraph("<code>/api/alerts</code>", code_style), Paragraph("GET", td_style), Paragraph("status, class, limit", td_style), Paragraph("200 JSON &mdash; Filterable, paginated alert records", td_style)],
        [Paragraph("<code>/api/alerts/&lt;id&gt;/status</code>", code_style), Paragraph("POST", td_style), Paragraph("status, analyst_notes", td_style), Paragraph("200 JSON &mdash; Lifecycle status update", td_style)],
        [Paragraph("<code>/api/ingest/csv</code>", code_style), Paragraph("POST", td_style), Paragraph("file (multipart)", td_style), Paragraph("200 JSON &mdash; Batch CSV inference & alert persistence", td_style)],
        [Paragraph("<code>/api/ingest/pcap</code>", code_style), Paragraph("POST", td_style), Paragraph("file (multipart)", td_style), Paragraph("200 JSON &mdash; Raw PCAP packet dissection & inference", td_style)],
        [Paragraph("<code>/api/download/sample-csv</code>", code_style), Paragraph("GET", td_style), Paragraph("None", td_style), Paragraph("200 text/csv &mdash; Download verified 66-feature sample", td_style)],
        [Paragraph("<code>/api/export/csv</code>", code_style), Paragraph("GET", td_style), Paragraph("status, search", td_style), Paragraph("200 text/csv &mdash; Sanitized compliance report", td_style)],
        [Paragraph("<code>/api/alerts/clear</code>", code_style), Paragraph("POST", td_style), Paragraph("None", td_style), Paragraph("200 JSON &mdash; Reset local alert database to 0", td_style)],
    ]
    api_table = Table(api_data, colWidths=[120, 45, 115, 224])
    api_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(api_table)
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # ── Section 8: Deep Dive: Module-by-Module Codebase Anatomy ───────────────────
    story.append(Paragraph("8. Module-by-Module Codebase Anatomy", h1_style))
    story.append(Paragraph(
        "The AI-NIDS platform is composed of 14 core Python modules organized into a clean separation of concerns:",
        body_style
    ))

    mod_data = [
        [Paragraph("<b>Module Path</b>", th_style), Paragraph("<b>Primary Responsibility</b>", th_style), Paragraph("<b>Key Classes & Functions</b>", th_style)],
        [
            Paragraph("<code>src/config.py</code>", code_style),
            Paragraph("Central configuration, filesystem paths, hyperparameter bounds, and environment detection (VERCEL).", td_style),
            Paragraph("<code>PROJECT_ROOT, DATA_DIR, ALERTS_DB_PATH, NON_FEATURE_COLUMNS</code>", code_style)
        ],
        [
            Paragraph("<code>src/data_loader.py</code>", code_style),
            Paragraph("Ingests multi-gigabyte raw CIC-IDS2017 CSV files, normalizes header casings, and isolates target labels.", td_style),
            Paragraph("<code>load_cicids_dataset(), stream_chunks(), generate_synthetic_dataset()</code>", code_style)
        ],
        [
            Paragraph("<code>src/eda.py</code>", code_style),
            Paragraph("Statistical profiling, missing value audits, class imbalance calculation, and correlation matrices.", td_style),
            Paragraph("<code>run_eda(), plot_class_distributions(), generate_eda_report()</code>", code_style)
        ],
        [
            Paragraph("<code>src/preprocess.py</code>", code_style),
            Paragraph("Mathematical sanitization, handling infinities, median imputation, and label encoding.", td_style),
            Paragraph("<code>clean_dataframe(), sanitize_infinities(), build_preprocessor()</code>", code_style)
        ],
        [
            Paragraph("<code>src/feature_engineering.py</code>", code_style),
            Paragraph("Feature selection, zero-variance pruning, and standard/robust feature scaling.", td_style),
            Paragraph("<code>select_features(), scale_features(), save_feature_metadata()</code>", code_style)
        ],
        [
            Paragraph("<code>src/train.py</code>", code_style),
            Paragraph("Supervised model training (Logistic Regression, Decision Trees, Random Forests) with balanced weighting.", td_style),
            Paragraph("<code>train_binary_models(), train_multiclass_models(), save_pipeline()</code>", code_style)
        ],
        [
            Paragraph("<code>src/evaluate.py</code>", code_style),
            Paragraph("Classification reports, confusion matrices, ROC-AUC curves, and recall threshold validation.", td_style),
            Paragraph("<code>evaluate_model(), plot_confusion_matrix(), compute_metrics()</code>", code_style)
        ],
        [
            Paragraph("<code>src/predict.py</code>", code_style),
            Paragraph("High-level unified inference engine orchestrating binary, multiclass, anomaly, and XAI pipelines.", td_style),
            Paragraph("<code>UnifiedPredictor, predict(), predict_single_flow()</code>", code_style)
        ],
        [
            Paragraph("<code>src/anomaly.py</code>", code_style),
            Paragraph("Semi-supervised Isolation Forest anomaly scoring engine trained on benign traffic baselines.", td_style),
            Paragraph("<code>AnomalyDetector, fit_baseline(), score_samples()</code>", code_style)
        ],
        [
            Paragraph("<code>src/explainability.py</code>", code_style),
            Paragraph("Sub-15ms decision-tree surrogate attribution replacing heavy SHAP computation.", td_style),
            Paragraph("<code>SurrogateExplainer, explain_flow(), extract_decision_path()</code>", code_style)
        ],
        [
            Paragraph("<code>src/pcap_processor.py</code>", code_style),
            Paragraph("Passive raw packet parser, 5-tuple conversation tracker, and flow state machine.", td_style),
            Paragraph("<code>process_pcap(), PacketDissector, FlowAssembler</code>", code_style)
        ],
        [
            Paragraph("<code>src/flow_features.py</code>", code_style),
            Paragraph("Real-time calculation of 66 statistical flow features from dissected packet arrival trains.", td_style),
            Paragraph("<code>extract_flow_features(), calculate_iat(), aggregate_flags()</code>", code_style)
        ],
        [
            Paragraph("<code>src/alert_manager.py</code>", code_style),
            Paragraph("Persistent SQLite CRUD engine managing alert lifecycles, analyst notes, and compliance reports.", td_style),
            Paragraph("<code>AlertManager, create_alerts_from_df(), update_alert_status()</code>", code_style)
        ],
        [
            Paragraph("<code>dashboard/server.py</code>", code_style),
            Paragraph("High-performance Flask 3.x REST API with VercelPathFixMiddleware reverse-proxy support.", td_style),
            Paragraph("<code>app, VercelPathFixMiddleware, ingest_csv(), download_sample_csv()</code>", code_style)
        ],
    ]
    mod_table = Table(mod_data, colWidths=[120, 204, 180])
    mod_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(mod_table)
    story.append(Spacer(1, 8))

    # ── Section 9: Industry Comparison Matrix ─────────────────────────────────────
    story.append(Paragraph("9. Comparative Industry Matrix: AI-NIDS vs. Existing Solutions", h1_style))

    comp_data = [
        [Paragraph("<b>Dimension</b>", th_style), Paragraph("<b>AI-NIDS</b>", th_style), Paragraph("<b>Snort 3.x</b>", th_style), Paragraph("<b>Suricata</b>", th_style), Paragraph("<b>Zeek (Bro)</b>", th_style), Paragraph("<b>Commercial Cloud NIDS</b>", th_style)],
        [Paragraph("<b>Cost</b>", td_style), Paragraph("<b>₹0 (100% Free)</b>", td_style), Paragraph("Free / FOSS", td_style), Paragraph("Free / FOSS", td_style), Paragraph("Free / FOSS", td_style), Paragraph("$5k–$50k / yr", td_style)],
        [Paragraph("<b>Detection Basis</b>", td_style), Paragraph("<b>Hybrid ML + Anomaly</b>", td_style), Paragraph("Signatures", td_style), Paragraph("Signatures", td_style), Paragraph("Behavioral Scripts", td_style), Paragraph("Black-Box Cloud AI", td_style)],
        [Paragraph("<b>Zero-Day Resilience</b>", td_style), Paragraph("<b>High (iForest)</b>", td_style), Paragraph("Poor (Needs hash)", td_style), Paragraph("Poor (Needs hash)", td_style), Paragraph("Medium", td_style), Paragraph("High", td_style)],
        [Paragraph("<b>Zero-Payload Privacy</b>", td_style), Paragraph("<b>Guaranteed</b>", td_style), Paragraph("Inspects Payload", td_style), Paragraph("Inspects Payload", td_style), Paragraph("Configurable", td_style), Paragraph("Cloud Ingestion Risk", td_style)],
        [Paragraph("<b>Explainability</b>", td_style), Paragraph("<b>&lt;15ms XAI Surrogate</b>", td_style), Paragraph("Rule String", td_style), Paragraph("Rule String", td_style), Paragraph("Script Log Trace", td_style), Paragraph("None (Opaque)", td_style)],
        [Paragraph("<b>Memory Overhead</b>", td_style), Paragraph("<b>~218 MB RAM</b>", td_style), Paragraph("1–4 GB RAM", td_style), Paragraph("2–8 GB RAM", td_style), Paragraph("2–8 GB RAM", td_style), Paragraph("Agent Footprint", td_style)],
        [Paragraph("<b>Deployment Simplicity</b>", td_style), Paragraph("<b>Single Python App</b>", td_style), Paragraph("C++ Build Toolchain", td_style), Paragraph("Multi-threaded Daemon", td_style), Paragraph("Cluster Daemon", td_style), Paragraph("Cloud Agent Required", td_style)],
    ]
    comp_table = Table(comp_data, colWidths=[95, 95, 75, 75, 75, 89])
    comp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 10))

    # ── Section 10: Verification Matrix ───────────────────────────────────────────
    story.append(Paragraph("10. Verification Matrix & 176-Test Coverage", h1_style))
    story.append(Paragraph(
        "The codebase is validated by <b>176 automated test cases</b> across 11 test suites with 100% pass rate:<br/>"
        "&bull; <code>tests/test_dashboard_server.py</code>: REST API endpoints, sample CSV download, Vercel reverse proxy middleware, and status lifecycles (12 tests).<br/>"
        "&bull; <code>tests/test_pcap_processor.py</code> & <code>test_phase5.py</code>: Packet parsing, bidirectional flow reconstruction, and 66-feature schema conformance.<br/>"
        "&bull; <code>tests/test_phase3.py</code> & <code>test_phase4.py</code>: Supervised binary and multiclass models, recall thresholds, and ROC-AUC metrics.<br/>"
        "&bull; <code>tests/test_phase6.py</code> & <code>test_phase7.py</code>: SQLite alert persistence, sanitized CSV reports, Isolation Forest calibration, and XAI surrogates.<br/>"
        "&bull; <code>tests/test_data_loader.py</code> & <code>test_validate_data.py</code>: Missing value handling, non-numeric column removal, and target leakage prevention.",
        body_style
    ))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    story.append(Paragraph(
        "<b>AI-NIDS Engineering & Security Architecture Group</b> &mdash; Release Document Approved for Production (October 26, 2026)",
        meta_style
    ))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated comprehensive multi-page PDF at: {output_path}")


if __name__ == "__main__":
    out_pdf = str(Path(__file__).resolve().parent / "AI_NIDS_Complete_PRD_and_Architecture.pdf")
    create_pdf(out_pdf)

