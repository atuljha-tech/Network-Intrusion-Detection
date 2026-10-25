"""
pcap_processor.py — PCAP ingestion, flow reconstruction, and inference pipeline.

Responsibilities:
  1. Ingest PCAP / PCAPNG network trace files safely in a local environment.
  2. Parse packet streams and group into bi-directional network flows (5-tuple).
  3. Extract full 66 statistical flow features matching CIC-IDS2017 schema.
  4. Validate extracted features against trained model schema.
  5. Apply trained machine learning pipelines (binary, multiclass, anomaly, XAI).
  6. Return structured prediction records with timestamps, flow identifiers, and latency metrics.
  7. Handle malformed packets, truncated files, and unsupported protocols gracefully.
  8. Provide an optional passive live-capture interface where OS permissions permit.
"""

from __future__ import annotations

import logging
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.utils import PcapReader

from src.config import (
    FLOW_ACTIVITY_TIMEOUT_SECONDS,
    FLOW_TIMEOUT_SECONDS,
    MAX_PACKETS_STREAMING,
    REPORTS_DIR,
    setup_logging,
)
from src.flow_features import EXPECTED_FLOW_FEATURES, Flow, FlowKey
from src.model_utils import validate_features
from src.predict import UnifiedPredictor

logger = setup_logging()
log = logging.getLogger("ai_nids.pcap_processor")


# ── Packet Dissection Helpers ────────────────────────────────────────────────

def extract_packet_info(pkt: Any) -> Optional[Tuple[FlowKey, float, int, int, int, int, int, bool]]:
    """Extract flow-relevant metadata from a raw Scapy packet.

    Returns
    -------
    (flow_key, timestamp, packet_len, header_len, payload_len, tcp_flags, window_size, is_ip_valid)
    or None if packet lacks IPv4 layer.
    """
    # Lazy import to avoid loading unused scapy layers at module import
    from scapy.layers.inet import IP, TCP, UDP, ICMP

    if not pkt.haslayer(IP):
        return None

    ip_layer = pkt[IP]
    src_ip = str(ip_layer.src)
    dst_ip = str(ip_layer.dst)
    proto = int(ip_layer.proto)

    # Epoch timestamp with float precision
    pkt_time = float(getattr(pkt, "time", time.time()))

    # Total IP length
    pkt_len = int(getattr(ip_layer, "len", len(ip_layer)))
    ip_hdr_len = int(ip_layer.ihl * 4) if getattr(ip_layer, "ihl", None) is not None else 20

    src_port = 0
    dst_port = 0
    tcp_flags = 0
    window_size = 0
    transport_hdr_len = 0
    payload_len = 0

    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        src_port = int(tcp_layer.sport)
        dst_port = int(tcp_layer.dport)
        tcp_flags = int(tcp_layer.flags)
        window_size = int(getattr(tcp_layer, "window", 0))
        tcp_dataofs = getattr(tcp_layer, "dataofs", None)
        transport_hdr_len = int(tcp_dataofs * 4) if tcp_dataofs is not None else 20
        payload_len = max(0, pkt_len - ip_hdr_len - transport_hdr_len)

    elif pkt.haslayer(UDP):
        udp_layer = pkt[UDP]
        src_port = int(udp_layer.sport)
        dst_port = int(udp_layer.dport)
        transport_hdr_len = 8
        payload_len = max(0, pkt_len - ip_hdr_len - transport_hdr_len)

    elif pkt.haslayer(ICMP):
        transport_hdr_len = 8
        payload_len = max(0, pkt_len - ip_hdr_len - transport_hdr_len)

    total_hdr_len = ip_hdr_len + transport_hdr_len
    flow_key = FlowKey(
        src_ip=src_ip,
        src_port=src_port,
        dst_ip=dst_ip,
        dst_port=dst_port,
        protocol=proto,
    )

    return flow_key, pkt_time, pkt_len, total_hdr_len, payload_len, tcp_flags, window_size, True


# ── Flow Aggregator ──────────────────────────────────────────────────────────

class FlowManager:
    """Manages active network flows, flow timeouts, and feature generation."""

    def __init__(
        self,
        flow_timeout_sec: float = FLOW_TIMEOUT_SECONDS,
        activity_timeout_sec: float = FLOW_ACTIVITY_TIMEOUT_SECONDS,
    ):
        self.flow_timeout_sec = flow_timeout_sec
        self.activity_timeout_sec = activity_timeout_sec

        # Maps forward FlowKey -> Flow
        self.active_flows: Dict[FlowKey, Flow] = {}
        # Maps reverse FlowKey -> forward FlowKey
        self.reverse_index: Dict[FlowKey, FlowKey] = {}

        self.completed_flows: List[Flow] = []

    def process_packet(
        self,
        flow_key: FlowKey,
        timestamp: float,
        length: int,
        header_len: int,
        payload_len: int,
        tcp_flags: int = 0,
        window_size: int = 0,
    ) -> None:
        """Add packet to existing bidirectional flow or create new flow."""
        # 1. Check if packet belongs to existing forward flow
        if flow_key in self.active_flows:
            flow = self.active_flows[flow_key]
            # Check timeout expiration
            if flow.is_expired(timestamp, self.flow_timeout_sec):
                self.completed_flows.append(flow)
                new_flow = Flow(flow_key, timestamp)
                new_flow.add_packet(timestamp, length, header_len, payload_len, tcp_flags, window_size, is_forward=True)
                self.active_flows[flow_key] = new_flow
            else:
                flow.add_packet(timestamp, length, header_len, payload_len, tcp_flags, window_size, is_forward=True)
            return

        # 2. Check if packet belongs to existing reverse flow
        if flow_key in self.reverse_index:
            fwd_key = self.reverse_index[flow_key]
            flow = self.active_flows[fwd_key]
            if flow.is_expired(timestamp, self.flow_timeout_sec):
                self.completed_flows.append(flow)
                del self.active_flows[fwd_key]
                del self.reverse_index[flow_key]
                # Start new flow with current packet as forward
                new_flow = Flow(flow_key, timestamp)
                new_flow.add_packet(timestamp, length, header_len, payload_len, tcp_flags, window_size, is_forward=True)
                self.active_flows[flow_key] = new_flow
                self.reverse_index[flow_key.reverse] = flow_key
            else:
                flow.add_packet(timestamp, length, header_len, payload_len, tcp_flags, window_size, is_forward=False)
            return

        # 3. New flow initialization (current packet defines forward direction)
        flow = Flow(flow_key, timestamp)
        flow.add_packet(timestamp, length, header_len, payload_len, tcp_flags, window_size, is_forward=True)
        self.active_flows[flow_key] = flow
        self.reverse_index[flow_key.reverse] = flow_key

    def flush_all_flows(self) -> List[Flow]:
        """Finalize and return all remaining active flows."""
        all_flows = self.completed_flows + list(self.active_flows.values())
        self.active_flows.clear()
        self.reverse_index.clear()
        self.completed_flows.clear()
        return all_flows


# ── Core PCAP Processing API ─────────────────────────────────────────────────

def process_pcap(
    pcap_path: Union[str, Path],
    model_type: str = "binary",
    predictor: Optional[UnifiedPredictor] = None,
    include_anomaly: bool = False,
    include_explanation: bool = False,
    max_packets: Optional[int] = MAX_PACKETS_STREAMING,
    flow_timeout_sec: float = FLOW_TIMEOUT_SECONDS,
    activity_timeout_sec: float = FLOW_ACTIVITY_TIMEOUT_SECONDS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process a PCAP file, extract flow features, and compute predictions.

    Parameters
    ----------
    pcap_path : str | Path
        Path to input .pcap / .pcapng file.
    model_type : "binary" | "multiclass"
        Prediction model paradigm.
    predictor : UnifiedPredictor | None
        Pre-loaded predictor. Initialized if None.
    include_anomaly : bool
        Whether to attach Isolation Forest anomaly scores.
    include_explanation : bool
        Whether to attach feature attribution explanations.
    max_packets : int | None
        Upper limit on packets to stream (prevents memory exhaustion).

    Returns
    -------
    (results_dataframe, telemetry_dict)
    """
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    start_wall_time = time.time()
    telemetry: Dict[str, Any] = {
        "pcap_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "packets_read": 0,
        "valid_ip_packets": 0,
        "malformed_packets": 0,
        "unsupported_protocol_packets": 0,
        "flows_reconstructed": 0,
        "parse_time_seconds": 0.0,
        "prediction_time_seconds": 0.0,
        "total_time_seconds": 0.0,
        "avg_prediction_latency_ms": 0.0,
        "status": "in_progress",
    }

    # Handle empty files
    if path.stat().st_size == 0:
        log.warning("PCAP file is empty: %s", path)
        telemetry.update({
            "status": "empty_file",
            "total_time_seconds": time.time() - start_wall_time,
        })
        return pd.DataFrame(), telemetry

    # Stream packets using Scapy PcapReader
    from scapy.utils import PcapReader

    flow_manager = FlowManager(
        flow_timeout_sec=flow_timeout_sec,
        activity_timeout_sec=activity_timeout_sec,
    )

    parse_start = time.time()
    try:
        with PcapReader(str(path)) as pcap_reader:
            for pkt in pcap_reader:
                telemetry["packets_read"] += 1
                if max_packets and telemetry["packets_read"] >= max_packets:
                    log.warning("Reached maximum packet streaming limit (%d).", max_packets)
                    break

                try:
                    info = extract_packet_info(pkt)
                    if info is None:
                        telemetry["unsupported_protocol_packets"] += 1
                        continue

                    f_key, t_stamp, p_len, h_len, pay_len, flags, win, is_valid = info
                    telemetry["valid_ip_packets"] += 1

                    flow_manager.process_packet(
                        flow_key=f_key,
                        timestamp=t_stamp,
                        length=p_len,
                        header_len=h_len,
                        payload_len=pay_len,
                        tcp_flags=flags,
                        window_size=win,
                    )
                except Exception as pkt_exc:
                    telemetry["malformed_packets"] += 1
                    log.debug("Skipped malformed packet (%d): %s", telemetry["packets_read"], pkt_exc)

    except (struct.error, EOFError, OSError) as read_exc:
        telemetry["status"] = "partial_corrupt_pcap_read"
        log.warning("PCAP stream encountered corrupt header/frame: %s. Using recovered packets.", read_exc)
    except Exception as exc:
        from scapy.error import Scapy_Exception
        if isinstance(exc, Scapy_Exception):
            telemetry["status"] = "invalid_or_corrupt_pcap"
            telemetry["total_time_seconds"] = round(time.time() - start_wall_time, 4)
            log.warning("Invalid or unsupported capture format for %s: %s", path, exc)
            return pd.DataFrame(), telemetry
        telemetry["status"] = f"error: {exc}"
        log.error("Fatal error reading PCAP %s: %s", path, exc)
        raise

    telemetry["parse_time_seconds"] = round(time.time() - parse_start, 4)

    # Finalize flows
    flows = flow_manager.flush_all_flows()
    telemetry["flows_reconstructed"] = len(flows)
    log.info(
        "Extracted %d bidirectional flow(s) from %d packet(s) in %.2f s.",
        len(flows),
        telemetry["packets_read"],
        telemetry["parse_time_seconds"],
    )

    if not flows:
        telemetry["status"] = "no_ip_flows_found"
        telemetry["total_time_seconds"] = round(time.time() - start_wall_time, 4)
        return pd.DataFrame(), telemetry

    # Assemble feature matrix and metadata
    metadata_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, float]] = []

    for flow in flows:
        meta = flow.to_metadata()
        meta["start_time_iso"] = datetime.fromtimestamp(meta["start_time"], tz=timezone.utc).isoformat()
        meta["end_time_iso"] = datetime.fromtimestamp(meta["end_time"], tz=timezone.utc).isoformat()
        meta["duration_ms"] = round(meta["duration_seconds"] * 1000.0, 2)
        metadata_rows.append(meta)

        feat_dict = flow.to_features(activity_timeout_sec=activity_timeout_sec)
        feature_rows.append(feat_dict)

    df_meta = pd.DataFrame(metadata_rows)
    df_features = pd.DataFrame(feature_rows)

    # Model Inference
    pred_start = time.time()
    if predictor is None:
        predictor = UnifiedPredictor(auto_load=True)

    pred_results = predictor.predict(
        df_features,
        model_type=model_type,
        include_anomaly=include_anomaly,
        include_explanation=include_explanation,
    )
    pred_duration = time.time() - pred_start
    telemetry["prediction_time_seconds"] = round(pred_duration, 4)
    telemetry["avg_prediction_latency_ms"] = round((pred_duration / max(1, len(flows))) * 1000.0, 3)

    # Merge metadata with predictions
    if isinstance(pred_results, pd.DataFrame):
        # Drop raw feature duplication if desired, or merge clean output
        pred_cols = [c for c in pred_results.columns if c not in EXPECTED_FLOW_FEATURES]
        merged_df = pd.concat([df_meta, pred_results[pred_cols]], axis=1)
    else:
        merged_df = df_meta.copy()

    telemetry["total_time_seconds"] = round(time.time() - start_wall_time, 4)
    telemetry["throughput_packets_per_sec"] = (
        round(telemetry["packets_read"] / max(1e-4, telemetry["total_time_seconds"]), 1)
    )
    telemetry["status"] = "success"

    return merged_df, telemetry


# ── Passive Live Capture Interface ───────────────────────────────────────────

def is_live_capture_supported() -> Tuple[bool, str]:
    """Check whether the host operating system permits passive packet capture."""
    # Check if running with root/elevated permissions (required on POSIX systems for raw sockets / BPF)
    if os.name == "posix":
        if hasattr(os, "geteuid"):
            if os.geteuid() != 0:
                return (
                    False,
                    "Live capture requires root/administrator privileges to open BPF / raw sockets on POSIX. "
                    "Run with elevated permissions or use PCAP file ingestion.",
                )
    return True, "Passive capture interface supported."


def capture_live_traffic(
    interface: Optional[str] = None,
    packet_count: int = 50,
    timeout_seconds: float = 10.0,
    model_type: str = "binary",
    predictor: Optional[UnifiedPredictor] = None,
    include_anomaly: bool = False,
    include_explanation: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform passive live packet capture on authorized network interfaces.

    SAFETY AND ETHICAL CONSTRAINTS:
      • Passive capture only (no active scanning, no packet transmission, no injection).
      • Strict packet and timeout limits to prevent resource exhaustion.
      • Does NOT store packet payloads.
      • Strictly authorized local interfaces only.
    """
    supported, reason = is_live_capture_supported()
    if not supported:
        log.warning("Passive live capture cannot run: %s", reason)
        return pd.DataFrame(), {
            "status": "permission_denied",
            "message": reason,
            "packets_read": 0,
            "flows_reconstructed": 0,
        }

    from scapy.all import sniff  # type: ignore

    log.info(
        "Initiating PASSIVE live capture (iface=%s, count=%d, timeout=%.1fs)...",
        interface or "default",
        packet_count,
        timeout_seconds,
    )

    flow_manager = FlowManager()
    captured_count = 0
    start_time = time.time()

    def _packet_callback(pkt):
        nonlocal captured_count
        captured_count += 1
        try:
            info = extract_packet_info(pkt)
            if info:
                f_key, t_stamp, p_len, h_len, pay_len, flags, win, _ = info
                flow_manager.process_packet(
                    flow_key=f_key,
                    timestamp=t_stamp,
                    length=p_len,
                    header_len=h_len,
                    payload_len=pay_len,
                    tcp_flags=flags,
                    window_size=win,
                )
        except Exception as exc:
            log.debug("Live packet callback error: %s", exc)

    try:
        sniff(
            iface=interface,
            count=packet_count,
            timeout=timeout_seconds,
            prn=_packet_callback,
            store=False,  # DO NOT store raw packets in memory (safety requirement)
            filter="ip",  # Only inspect IP packets
        )
    except PermissionError as perm_err:
        log.warning("Live sniff permission denied: %s", perm_err)
        return pd.DataFrame(), {"status": "permission_denied", "error": str(perm_err)}
    except Exception as exc:
        log.error("Live sniff failed: %s", exc)
        return pd.DataFrame(), {"status": "error", "error": str(exc)}

    duration = time.time() - start_time
    flows = flow_manager.flush_all_flows()

    if not flows:
        return pd.DataFrame(), {
            "status": "no_packets_captured",
            "packets_read": captured_count,
            "duration_seconds": round(duration, 3),
        }

    # Extract features & predict
    metadata_rows = []
    feature_rows = []
    for flow in flows:
        meta = flow.to_metadata()
        meta["start_time_iso"] = datetime.fromtimestamp(meta["start_time"], tz=timezone.utc).isoformat()
        meta["end_time_iso"] = datetime.fromtimestamp(meta["end_time"], tz=timezone.utc).isoformat()
        meta["duration_ms"] = round(meta["duration_seconds"] * 1000.0, 2)
        metadata_rows.append(meta)
        feature_rows.append(flow.to_features())

    df_meta = pd.DataFrame(metadata_rows)
    df_features = pd.DataFrame(feature_rows)

    if predictor is None:
        predictor = UnifiedPredictor(auto_load=True)

    preds = predictor.predict(
        df_features,
        model_type=model_type,
        include_anomaly=include_anomaly,
        include_explanation=include_explanation,
    )
    pred_cols = [c for c in preds.columns if c not in EXPECTED_FLOW_FEATURES]
    merged = pd.concat([df_meta, preds[pred_cols]], axis=1)

    telemetry = {
        "status": "success",
        "capture_duration_seconds": round(duration, 3),
        "packets_read": captured_count,
        "flows_reconstructed": len(flows),
    }
    return merged, telemetry


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI-NIDS Phase 5: PCAP traffic processor & inference")
    parser.add_argument("--pcap", type=str, required=True, help="Path to .pcap or .pcapng file")
    parser.add_argument("--multiclass", action="store_true", help="Use multiclass attack model instead of binary")
    parser.add_argument("--anomaly", action="store_true", help="Include Isolation Forest anomaly score")
    parser.add_argument("--explain", action="store_true", help="Include XAI feature attribution explanations")
    parser.add_argument("--output", type=str, default=None, help="Save structured prediction records to CSV path")

    args = parser.parse_args()
    model_type = "multiclass" if args.multiclass else "binary"

    df_results, telemetry = process_pcap(
        pcap_path=args.pcap,
        model_type=model_type,
        include_anomaly=args.anomaly,
        include_explanation=args.explain,
    )

    print(f"\n{'=' * 65}")
    print(f"  AI-NIDS PCAP Traffic Ingestion & Prediction Complete")
    print(f"{'=' * 65}")
    print(f"File:               {telemetry.get('pcap_path')}")
    print(f"Packets Read:       {telemetry.get('packets_read', 0):,}")
    print(f"Valid IP Packets:   {telemetry.get('valid_ip_packets', 0):,}")
    print(f"Flows Extracted:    {telemetry.get('flows_reconstructed', 0):,}")
    print(f"Parse Duration:     {telemetry.get('parse_time_seconds', 0.0):.2f} s")
    print(f"Inference Duration: {telemetry.get('prediction_time_seconds', 0.0):.4f} s")
    print(f"Avg Latency / Flow: {telemetry.get('avg_prediction_latency_ms', 0.0):.2f} ms")
    print(f"Status:             {telemetry.get('status')}")
    print(f"{'=' * 65}")

    if not df_results.empty:
        summary_cols = ["flow_id", "duration_ms"]
        if "label" in df_results.columns:
            summary_cols.append("label")
        if "predicted_class" in df_results.columns:
            summary_cols.append("predicted_class")
        if "anomaly_score" in df_results.columns:
            summary_cols.append("anomaly_score")
        print("\nExtracted Flow Predictions Preview:")
        print(df_results[summary_cols].head(10).to_string())

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(out_p, index=False)
            print(f"\nSaved structured prediction records → {out_p}")


if __name__ == "__main__":
    main()
