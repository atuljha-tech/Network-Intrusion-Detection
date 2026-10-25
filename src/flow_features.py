"""
flow_features.py — Bi-directional network flow accumulator and feature extractor.

Computes the 66 statistical network flow features matching the exact CIC-IDS2017
feature schema used by the AI-NIDS trained models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ai_nids.flow_features")

# Standard CIC-IDS2017 66-feature schema in exact expected order
EXPECTED_FLOW_FEATURES: List[str] = [
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    "fwd_header_length",
    "bwd_header_length",
    "fwd_packets_s",
    "bwd_packets_s",
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "cwe_flag_count",
    "ece_flag_count",
    "down_up_ratio",
    "average_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "init_win_bytes_forward",
    "init_win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
]


@dataclass(frozen=True)
class FlowKey:
    """Bi-directional flow identifier (5-tuple)."""
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: int

    @property
    def reverse(self) -> FlowKey:
        """Return the reverse flow key (for backward direction)."""
        return FlowKey(
            src_ip=self.dst_ip,
            src_port=self.dst_port,
            dst_ip=self.src_ip,
            dst_port=self.src_port,
            protocol=self.protocol,
        )

    def __str__(self) -> str:
        proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(self.protocol, str(self.protocol))
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}({proto_name})"


@dataclass
class PacketRecord:
    """Internal lightweight representation of a packet within a flow."""
    timestamp: float        # Epoch seconds (float)
    length: int             # Total IP packet length in bytes
    header_length: int      # Header length (IP + transport) in bytes
    payload_length: int     # Transport payload length in bytes
    tcp_flags: int          # TCP flags bitmask (0 for UDP/ICMP)
    window_size: int        # TCP window size (0 for UDP/ICMP)


class Flow:
    """Accumulates packets for a bidirectional flow and calculates statistical features."""

    def __init__(self, key: FlowKey, first_packet_time: float):
        self.key = key
        self.first_time = first_packet_time
        self.last_time = first_packet_time

        self.fwd_packets: List[PacketRecord] = []
        self.bwd_packets: List[PacketRecord] = []

    def add_packet(
        self,
        timestamp: float,
        length: int,
        header_length: Optional[int] = None,
        payload_length: int = 0,
        tcp_flags: int = 0,
        window_size: int = 0,
        is_forward: bool = True,
        header_len: Optional[int] = None,
    ) -> None:
        """Add a packet to this flow in the forward or backward direction."""
        actual_header_len = header_length if header_length is not None else (header_len if header_len is not None else 20)
        record = PacketRecord(
            timestamp=timestamp,
            length=length,
            header_length=actual_header_len,
            payload_length=payload_length,
            tcp_flags=tcp_flags,
            window_size=window_size,
        )

        if is_forward:
            self.fwd_packets.append(record)
        else:
            self.bwd_packets.append(record)

        if timestamp > self.last_time:
            self.last_time = timestamp
        if timestamp < self.first_time:
            self.first_time = timestamp

    @property
    def total_packets(self) -> int:
        return len(self.fwd_packets) + len(self.bwd_packets)

    def is_expired(self, current_time: float, timeout_seconds: float = 120.0) -> bool:
        """Check if flow has been inactive longer than timeout."""
        return (current_time - self.last_time) >= timeout_seconds

    def to_metadata(self) -> Dict[str, Any]:
        """Return operational metadata identifying this flow."""
        duration_sec = max(0.0, self.last_time - self.first_time)
        return {
            "flow_id": str(self.key),
            "src_ip": self.key.src_ip,
            "src_port": self.key.src_port,
            "dst_ip": self.key.dst_ip,
            "dst_port": self.key.dst_port,
            "protocol": self.key.protocol,
            "start_time": self.first_time,
            "end_time": self.last_time,
            "duration_seconds": duration_sec,
            "total_fwd_packets": len(self.fwd_packets),
            "total_bwd_packets": len(self.bwd_packets),
        }

    def to_features(self, activity_timeout_sec: float = 5.0) -> Dict[str, float]:
        """Compute the full 66 statistical flow features matching CIC-IDS2017 schema."""
        all_pkts = sorted(
            [(p, True) for p in self.fwd_packets] + [(p, False) for p in self.bwd_packets],
            key=lambda x: x[0].timestamp,
        )

        n_fwd = len(self.fwd_packets)
        n_bwd = len(self.bwd_packets)
        n_total = n_fwd + n_bwd

        # 1. Flow Duration (microseconds)
        duration_sec = max(0.0, self.last_time - self.first_time)
        duration_us = duration_sec * 1_000_000.0

        # 2. Packet Lengths
        fwd_lengths = [p.length for p in self.fwd_packets]
        bwd_lengths = [p.length for p in self.bwd_packets]
        all_lengths = [p.length for p, _ in all_pkts]

        total_len_fwd = float(sum(fwd_lengths))
        total_len_bwd = float(sum(bwd_lengths))

        fwd_len_max = float(max(fwd_lengths)) if fwd_lengths else 0.0
        fwd_len_min = float(min(fwd_lengths)) if fwd_lengths else 0.0
        fwd_len_mean = float(np.mean(fwd_lengths)) if fwd_lengths else 0.0
        fwd_len_std = float(np.std(fwd_lengths)) if len(fwd_lengths) > 1 else 0.0

        bwd_len_max = float(max(bwd_lengths)) if bwd_lengths else 0.0
        bwd_len_min = float(min(bwd_lengths)) if bwd_lengths else 0.0
        bwd_len_mean = float(np.mean(bwd_lengths)) if bwd_lengths else 0.0
        bwd_len_std = float(np.std(bwd_lengths)) if len(bwd_lengths) > 1 else 0.0

        min_pkt_len = float(min(all_lengths)) if all_lengths else 0.0
        max_pkt_len = float(max(all_lengths)) if all_lengths else 0.0
        pkt_len_mean = float(np.mean(all_lengths)) if all_lengths else 0.0
        pkt_len_std = float(np.std(all_lengths)) if len(all_lengths) > 1 else 0.0
        pkt_len_var = float(np.var(all_lengths)) if len(all_lengths) > 1 else 0.0

        # 3. Rates
        denom_sec = max(duration_sec, 1e-6)
        flow_bytes_s = (total_len_fwd + total_len_bwd) / denom_sec
        flow_packets_s = n_total / denom_sec
        fwd_packets_s = n_fwd / denom_sec
        bwd_packets_s = n_bwd / denom_sec

        # 4. Inter-Arrival Times (IAT in microseconds)
        def _compute_iats(timestamps: List[float]) -> List[float]:
            if len(timestamps) < 2:
                return []
            return [(t2 - t1) * 1_000_000.0 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]

        all_timestamps = [p.timestamp for p, _ in all_pkts]
        fwd_timestamps = [p.timestamp for p in self.fwd_packets]
        bwd_timestamps = [p.timestamp for p in self.bwd_packets]

        all_iats = _compute_iats(all_timestamps)
        fwd_iats = _compute_iats(fwd_timestamps)
        bwd_iats = _compute_iats(bwd_timestamps)

        flow_iat_mean = float(np.mean(all_iats)) if all_iats else 0.0
        flow_iat_std = float(np.std(all_iats)) if len(all_iats) > 1 else 0.0
        flow_iat_max = float(max(all_iats)) if all_iats else 0.0
        flow_iat_min = float(min(all_iats)) if all_iats else 0.0

        fwd_iat_tot = float(sum(fwd_iats))
        fwd_iat_mean = float(np.mean(fwd_iats)) if fwd_iats else 0.0
        fwd_iat_std = float(np.std(fwd_iats)) if len(fwd_iats) > 1 else 0.0
        fwd_iat_max = float(max(fwd_iats)) if fwd_iats else 0.0
        fwd_iat_min = float(min(fwd_iats)) if fwd_iats else 0.0

        bwd_iat_tot = float(sum(bwd_iats))
        bwd_iat_mean = float(np.mean(bwd_iats)) if bwd_iats else 0.0
        bwd_iat_std = float(np.std(bwd_iats)) if len(bwd_iats) > 1 else 0.0
        bwd_iat_max = float(max(bwd_iats)) if bwd_iats else 0.0
        bwd_iat_min = float(min(bwd_iats)) if bwd_iats else 0.0

        # 5. TCP Flags
        fin_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x01)
        syn_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x02)
        rst_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x04)
        psh_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x08)
        ack_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x10)
        urg_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x20)
        ece_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x40)
        cwe_count = sum(1 for p, _ in all_pkts if p.tcp_flags & 0x80)

        fwd_psh_flags = sum(1 for p in self.fwd_packets if p.tcp_flags & 0x08)
        bwd_psh_flags = sum(1 for p in self.bwd_packets if p.tcp_flags & 0x08)
        fwd_urg_flags = sum(1 for p in self.fwd_packets if p.tcp_flags & 0x20)
        bwd_urg_flags = sum(1 for p in self.bwd_packets if p.tcp_flags & 0x20)

        # 6. Header lengths and segment sizes
        fwd_header_len = float(sum(p.header_length for p in self.fwd_packets))
        bwd_header_len = float(sum(p.header_length for p in self.bwd_packets))
        min_seg_size_fwd = float(min([p.header_length for p in self.fwd_packets])) if self.fwd_packets else 0.0

        # 7. Ratios and Segment Averages
        down_up_ratio = float(n_bwd / max(1, n_fwd))
        avg_packet_size = (total_len_fwd + total_len_bwd) / max(1, n_total)
        avg_fwd_segment_size = fwd_len_mean
        avg_bwd_segment_size = bwd_len_mean

        # 8. TCP Initial Window Bytes
        init_win_fwd = float(self.fwd_packets[0].window_size) if self.fwd_packets else 0.0
        init_win_bwd = float(self.bwd_packets[0].window_size) if self.bwd_packets else 0.0

        # 9. Forward Data Packets (payload > 0)
        act_data_pkt_fwd = float(sum(1 for p in self.fwd_packets if p.payload_length > 0))

        # 10. Active and Idle times (burst analysis with activity timeout)
        active_times: List[float] = []
        idle_times: List[float] = []

        if len(all_timestamps) >= 2:
            current_active_start = all_timestamps[0]
            last_timestamp = all_timestamps[0]

            for t in all_timestamps[1:]:
                diff = t - last_timestamp
                if diff >= activity_timeout_sec:
                    # Previous burst becomes an active period
                    active_dur = (last_timestamp - current_active_start) * 1_000_000.0
                    active_times.append(max(active_dur, 0.0))
                    # Gap becomes idle period
                    idle_times.append(diff * 1_000_000.0)
                    current_active_start = t
                last_timestamp = t

            # Final active burst
            active_dur = (last_timestamp - current_active_start) * 1_000_000.0
            active_times.append(max(active_dur, 0.0))
        elif len(all_timestamps) == 1:
            active_times = [0.0]
            idle_times = [0.0]

        active_mean = float(np.mean(active_times)) if active_times else 0.0
        active_std = float(np.std(active_times)) if len(active_times) > 1 else 0.0
        active_max = float(max(active_times)) if active_times else 0.0
        active_min = float(min(active_times)) if active_times else 0.0

        idle_mean = float(np.mean(idle_times)) if idle_times else 0.0
        idle_std = float(np.std(idle_times)) if len(idle_times) > 1 else 0.0
        idle_max = float(max(idle_times)) if idle_times else 0.0
        idle_min = float(min(idle_times)) if idle_times else 0.0

        features = {
            "flow_duration": duration_us,
            "total_fwd_packets": float(n_fwd),
            "total_backward_packets": float(n_bwd),
            "total_length_of_fwd_packets": total_len_fwd,
            "total_length_of_bwd_packets": total_len_bwd,
            "fwd_packet_length_max": fwd_len_max,
            "fwd_packet_length_min": fwd_len_min,
            "fwd_packet_length_mean": fwd_len_mean,
            "fwd_packet_length_std": fwd_len_std,
            "bwd_packet_length_max": bwd_len_max,
            "bwd_packet_length_min": bwd_len_min,
            "bwd_packet_length_mean": bwd_len_mean,
            "bwd_packet_length_std": bwd_len_std,
            "flow_bytes_s": flow_bytes_s,
            "flow_packets_s": flow_packets_s,
            "flow_iat_mean": flow_iat_mean,
            "flow_iat_std": flow_iat_std,
            "flow_iat_max": flow_iat_max,
            "flow_iat_min": flow_iat_min,
            "fwd_iat_total": fwd_iat_tot,
            "fwd_iat_mean": fwd_iat_mean,
            "fwd_iat_std": fwd_iat_std,
            "fwd_iat_max": fwd_iat_max,
            "fwd_iat_min": fwd_iat_min,
            "bwd_iat_total": bwd_iat_tot,
            "bwd_iat_mean": bwd_iat_mean,
            "bwd_iat_std": bwd_iat_std,
            "bwd_iat_max": bwd_iat_max,
            "bwd_iat_min": bwd_iat_min,
            "fwd_psh_flags": float(fwd_psh_flags),
            "bwd_psh_flags": float(bwd_psh_flags),
            "fwd_urg_flags": float(fwd_urg_flags),
            "bwd_urg_flags": float(bwd_urg_flags),
            "fwd_header_length": fwd_header_len,
            "bwd_header_length": bwd_header_len,
            "fwd_packets_s": fwd_packets_s,
            "bwd_packets_s": bwd_packets_s,
            "min_packet_length": min_pkt_len,
            "max_packet_length": max_pkt_len,
            "packet_length_mean": pkt_len_mean,
            "packet_length_std": pkt_len_std,
            "packet_length_variance": pkt_len_var,
            "fin_flag_count": float(fin_count),
            "syn_flag_count": float(syn_count),
            "rst_flag_count": float(rst_count),
            "psh_flag_count": float(psh_count),
            "ack_flag_count": float(ack_count),
            "urg_flag_count": float(urg_count),
            "cwe_flag_count": float(cwe_count),
            "ece_flag_count": float(ece_count),
            "down_up_ratio": down_up_ratio,
            "average_packet_size": avg_packet_size,
            "avg_fwd_segment_size": avg_fwd_segment_size,
            "avg_bwd_segment_size": avg_bwd_segment_size,
            "init_win_bytes_forward": init_win_fwd,
            "init_win_bytes_backward": init_win_bwd,
            "act_data_pkt_fwd": act_data_pkt_fwd,
            "min_seg_size_forward": min_seg_size_fwd,
            "active_mean": active_mean,
            "active_std": active_std,
            "active_max": active_max,
            "active_min": active_min,
            "idle_mean": idle_mean,
            "idle_std": idle_std,
            "idle_max": idle_max,
            "idle_min": idle_min,
        }
        return features
