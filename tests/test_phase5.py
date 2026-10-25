"""
test_phase5.py — Unit and integration tests for AI-NIDS Phase 5 (Network Traffic Integration).

Tests:
  1. Synthetic PCAP generation and dissection (TCP handshake, data exchange, UDP).
  2. FlowAccumulator bidirectional tracking and feature calculation (all 66 features).
  3. Feature schema validation against trained model metadata.
  4. End-to-end PCAP processing through binary and multiclass models.
  5. Anomaly scoring and explainability integration on PCAP flows.
  6. Edge cases: empty PCAPs, malformed PCAPs, non-IP protocols (ARP), incomplete flows.
  7. Performance and latency telemetry measurements.
  8. Passive live-capture interface permission checks and safety guardrails.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether
from scapy.packet import Raw
from scapy.utils import wrpcap

from src.flow_features import EXPECTED_FLOW_FEATURES, Flow, FlowKey
from src.pcap_processor import (
    FlowManager,
    extract_packet_info,
    is_live_capture_supported,
    process_pcap,
)
from src.predict import UnifiedPredictor


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_tcp_pcap(tmp_path: Path) -> Path:
    """Generate a valid, legally compliant synthetic TCP three-way handshake and data exchange."""
    pcap_path = tmp_path / "valid_tcp.pcap"
    packets = []
    base_time = 1700000000.0

    # 1. SYN (Client -> Server)
    pkt1 = IP(src="192.168.1.100", dst="192.168.1.1") / TCP(sport=54321, dport=80, flags="S", seq=1000, window=64240)
    pkt1.time = base_time
    packets.append(pkt1)

    # 2. SYN-ACK (Server -> Client)
    pkt2 = IP(src="192.168.1.1", dst="192.168.1.100") / TCP(sport=80, dport=54321, flags="SA", seq=2000, ack=1001, window=65535)
    pkt2.time = base_time + 0.010
    packets.append(pkt2)

    # 3. ACK (Client -> Server)
    pkt3 = IP(src="192.168.1.100", dst="192.168.1.1") / TCP(sport=54321, dport=80, flags="A", seq=1001, ack=2001, window=64240)
    pkt3.time = base_time + 0.012
    packets.append(pkt3)

    # 4. HTTP GET (Client -> Server) with payload
    pkt4 = (
        IP(src="192.168.1.100", dst="192.168.1.1")
        / TCP(sport=54321, dport=80, flags="PA", seq=1001, ack=2001, window=64240)
        / Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n")
    )
    pkt4.time = base_time + 0.015
    packets.append(pkt4)

    # 5. HTTP 200 OK (Server -> Client) with payload
    pkt5 = (
        IP(src="192.168.1.1", dst="192.168.1.100")
        / TCP(sport=80, dport=54321, flags="PA", seq=2001, ack=1045, window=65535)
        / Raw(load=b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nHello World!")
    )
    pkt5.time = base_time + 0.025
    packets.append(pkt5)

    # 6. FIN-ACK (Client -> Server)
    pkt6 = IP(src="192.168.1.100", dst="192.168.1.1") / TCP(sport=54321, dport=80, flags="FA", seq=1045, ack=2045, window=64240)
    pkt6.time = base_time + 0.030
    packets.append(pkt6)

    # 7. ACK (Server -> Client)
    pkt7 = IP(src="192.168.1.1", dst="192.168.1.100") / TCP(sport=80, dport=54321, flags="A", seq=2045, ack=1046, window=65535)
    pkt7.time = base_time + 0.032
    packets.append(pkt7)

    wrpcap(str(pcap_path), packets)
    return pcap_path


@pytest.fixture
def synthetic_udp_pcap(tmp_path: Path) -> Path:
    """Generate a synthetic UDP request/response flow (e.g., DNS)."""
    pcap_path = tmp_path / "valid_udp.pcap"
    packets = []
    base_time = 1700000100.0

    # UDP Query
    p1 = IP(src="10.0.0.5", dst="10.0.0.1") / UDP(sport=43210, dport=53) / Raw(load=b"\x12\x34\x01\x00\x00\x01")
    p1.time = base_time
    packets.append(p1)

    # UDP Response
    p2 = IP(src="10.0.0.1", dst="10.0.0.5") / UDP(sport=53, dport=43210) / Raw(load=b"\x12\x34\x81\x80\x00\x01\x00\x01")
    p2.time = base_time + 0.005
    packets.append(p2)

    wrpcap(str(pcap_path), packets)
    return pcap_path


@pytest.fixture
def mixed_protocols_pcap(tmp_path: Path) -> Path:
    """Generate a PCAP containing non-IP (ARP), ICMP, and TCP packets with consistent linktype."""
    pcap_path = tmp_path / "mixed.pcap"
    packets = [
        # Non-IP packet (ARP)
        Ether() / ARP(pdst="192.168.1.1"),
        # ICMP packet
        Ether() / IP(src="192.168.1.50", dst="192.168.1.1") / ICMP(),
        # TCP packet
        Ether() / IP(src="192.168.1.50", dst="192.168.1.1") / TCP(sport=1111, dport=22, flags="S"),
    ]
    for idx, p in enumerate(packets):
        p.time = 1700000200.0 + idx
    wrpcap(str(pcap_path), packets)
    return pcap_path


# ── Flow Features & Accumulator Tests (Tasks 3, 5, 6) ────────────────────────

def test_flow_key_and_reverse():
    key = FlowKey(src_ip="1.1.1.1", src_port=1234, dst_ip="2.2.2.2", dst_port=80, protocol=6)
    rev = key.reverse

    assert rev.src_ip == "2.2.2.2"
    assert rev.src_port == 80
    assert rev.dst_ip == "1.1.1.1"
    assert rev.dst_port == 1234
    assert rev.protocol == 6
    assert rev.reverse == key


def test_flow_accumulation_and_feature_completeness():
    key = FlowKey(src_ip="10.0.0.1", src_port=50000, dst_ip="10.0.0.2", dst_port=443, protocol=6)
    flow = Flow(key, first_packet_time=100.0)

    # Forward SYN
    flow.add_packet(timestamp=100.0, length=60, header_len=40, payload_length=0, tcp_flags=0x02, window_size=65535, is_forward=True)
    # Backward SYN-ACK
    flow.add_packet(timestamp=100.02, length=60, header_len=40, payload_length=0, tcp_flags=0x12, window_size=65535, is_forward=False)
    # Forward PSH-ACK with data
    flow.add_packet(timestamp=100.05, length=200, header_len=40, payload_length=160, tcp_flags=0x18, window_size=65535, is_forward=True)

    features = flow.to_features()
    metadata = flow.to_metadata()

    # Exact 66 features must be present
    assert len(features) == 66
    for f in EXPECTED_FLOW_FEATURES:
        assert f in features, f"Missing feature: {f}"
        assert not np.isnan(features[f]), f"Feature {f} is NaN"
        assert not np.isinf(features[f]), f"Feature {f} is infinite"

    # Specific mathematical checks
    assert features["total_fwd_packets"] == 2.0
    assert features["total_backward_packets"] == 1.0
    assert features["flow_duration"] == pytest.approx(50000.0, abs=1.0)  # 0.05s = 50,000 us
    assert features["act_data_pkt_fwd"] == 1.0  # 1 forward packet with payload > 0
    assert features["syn_flag_count"] == 2.0   # SYN on fwd, SYN-ACK on bwd
    assert features["psh_flag_count"] == 1.0   # PSH on 3rd packet
    assert features["init_win_bytes_forward"] == 65535.0
    assert features["init_win_bytes_backward"] == 65535.0

    # Metadata checks
    assert metadata["src_ip"] == "10.0.0.1"
    assert metadata["dst_ip"] == "10.0.0.2"
    assert metadata["total_fwd_packets"] == 2
    assert metadata["total_bwd_packets"] == 1


def test_packet_info_extraction():
    p = IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=9999, dport=8080, flags="S", window=14600)
    p.time = 1234567.89
    p_bytes = IP(bytes(p))
    p_bytes.time = 1234567.89

    info = extract_packet_info(p_bytes)
    assert info is not None
    flow_key, pkt_time, pkt_len, total_hdr_len, payload_len, tcp_flags, window_size, is_valid = info

    assert flow_key.src_ip == "192.168.1.1"
    assert flow_key.dst_ip == "192.168.1.2"
    assert flow_key.src_port == 9999
    assert flow_key.dst_port == 8080
    assert flow_key.protocol == 6
    assert pkt_time == 1234567.89
    assert tcp_flags == 0x02
    assert window_size == 14600


# ── PCAP Processing and Inference (Tasks 2, 7, 8, 13) ───────────────────────

def test_process_pcap_end_to_end_binary(synthetic_tcp_pcap):
    results_df, telemetry = process_pcap(
        synthetic_tcp_pcap,
        model_type="binary",
        include_anomaly=True,
        include_explanation=True,
    )

    # Telemetry checks
    assert telemetry["status"] == "success"
    assert telemetry["packets_read"] == 7
    assert telemetry["valid_ip_packets"] == 7
    assert telemetry["flows_reconstructed"] == 1
    assert telemetry["parse_time_seconds"] >= 0.0
    assert telemetry["prediction_time_seconds"] >= 0.0
    assert telemetry["avg_prediction_latency_ms"] >= 0.0

    # DataFrame schema checks
    assert len(results_df) == 1
    row = results_df.iloc[0]

    assert row["src_ip"] == "192.168.1.100"
    assert row["dst_ip"] == "192.168.1.1"
    assert row["src_port"] == 54321
    assert row["dst_port"] == 80
    assert row["protocol"] == 6
    assert "start_time_iso" in row
    assert "end_time_iso" in row
    assert "duration_ms" in row

    # Prediction outputs
    assert "prediction" in row
    assert "label" in row
    assert row["label"] in ["Benign", "Malicious"]

    # Anomaly output
    assert "is_anomaly" in row
    assert "anomaly_score" in row
    assert 0.0 <= row["anomaly_score"] <= 1.0

    # Explanation output
    assert "explanation" in row
    assert isinstance(row["explanation"], dict)
    assert "top_features" in row["explanation"]


def test_process_pcap_multiclass(synthetic_tcp_pcap):
    results_df, telemetry = process_pcap(
        synthetic_tcp_pcap,
        model_type="multiclass",
        include_anomaly=False,
    )

    assert telemetry["status"] == "success"
    assert len(results_df) == 1
    assert "predicted_class" in results_df.columns
    assert "score" in results_df.columns


def test_process_udp_pcap(synthetic_udp_pcap):
    results_df, telemetry = process_pcap(
        synthetic_udp_pcap,
        model_type="binary",
    )

    assert telemetry["status"] == "success"
    assert telemetry["packets_read"] == 2
    assert telemetry["flows_reconstructed"] == 1
    assert results_df.iloc[0]["protocol"] == 17
    assert "prediction" in results_df.columns


# ── Edge Cases & Error Handling (Task 9) ─────────────────────────────────────

def test_missing_pcap_raises_error():
    with pytest.raises(FileNotFoundError):
        process_pcap("non_existent_file.pcap")


def test_empty_pcap_file(tmp_path: Path):
    empty_pcap = tmp_path / "empty.pcap"
    empty_pcap.write_bytes(b"")

    results_df, telemetry = process_pcap(empty_pcap)
    assert telemetry["status"] == "empty_file"
    assert len(results_df) == 0


def test_malformed_corrupt_pcap(tmp_path: Path):
    corrupt_pcap = tmp_path / "corrupt.pcap"
    corrupt_pcap.write_bytes(b"\xd4\xc3\xb2\xa1\x02\x00\x04\x00TRUNCATED_RANDOM_CORRUPT_BYTES_XYZ")

    results_df, telemetry = process_pcap(corrupt_pcap)
    # Must not crash; handles error gracefully
    assert telemetry["status"] in ["partial_corrupt_pcap_read", "no_ip_flows_found"]


def test_mixed_protocols_and_non_ip(mixed_protocols_pcap):
    results_df, telemetry = process_pcap(mixed_protocols_pcap)
    assert telemetry["status"] == "success"
    assert telemetry["packets_read"] == 3
    # Non-IP ARP packet correctly tracked
    assert telemetry["unsupported_protocol_packets"] == 1
    # IP packets (ICMP + TCP) processed
    assert telemetry["valid_ip_packets"] == 2


# ── Passive Live Capture Safety (Task 14) ───────────────────────────────────

def test_live_capture_supported_check():
    supported, msg = is_live_capture_supported()
    assert isinstance(supported, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_live_capture_safety_guardrails():
    """Ensure capture_live_traffic respects passive constraints and error handling."""
    from src.pcap_processor import capture_live_traffic

    # If non-root on POSIX, it must safely report permission_denied without crashing
    df, tel = capture_live_traffic(packet_count=5, timeout_seconds=1.0)
    assert isinstance(df, pd.DataFrame)
    assert "status" in tel
    assert tel["status"] in ["permission_denied", "success", "no_packets_captured"]
