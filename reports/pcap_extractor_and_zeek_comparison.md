# PCAP Feature Extraction, Schema Compatibility & Zeek Comparison

## 1. Executive Summary

Phase 5 of AI-NIDS bridges the gap between raw network traffic (PCAP / live packets) and machine learning inference. A critical engineering question is:

> *Can network monitoring tools like Zeek (formerly Bro) provide the exact features expected by models trained on CIC-IDS2017 / CICFlowMeter datasets?*

This document provides a technical comparison between **Zeek logs**, **CICFlowMeter**, and our native **Scapy flow extractor** ([`src/flow_features.py`](file:///Users/it4/Desktop/NetworkIntrusionDetection/src/flow_features.py)), documenting why standard Zeek connection logs cannot directly feed the existing 66-feature model without custom scripting, and detailing the exact traffic-to-feature pipeline.

---

## 2. Model Feature Schema Requirements

The trained AI-NIDS model pipelines ([`binary_intrusion_pipeline.joblib`](file:///Users/it4/Desktop/NetworkIntrusionDetection/models/binary_intrusion_pipeline.joblib) and [`multiclass_intrusion_pipeline.joblib`](file:///Users/it4/Desktop/NetworkIntrusionDetection/models/multiclass_intrusion_pipeline.joblib)) strictly require **66 statistical flow features**:

| Feature Category | Features (66 total) | Units / Representation |
|---|---|---|
| **Flow Timing** | `flow_duration`, `flow_bytes_s`, `flow_packets_s`, `fwd_packets_s`, `bwd_packets_s` | Duration in $\mu s$, rates per second |
| **Packet Lengths (Fwd)** | `total_length_of_fwd_packets`, `fwd_packet_length_max`, `fwd_packet_length_min`, `fwd_packet_length_mean`, `fwd_packet_length_std`, `avg_fwd_segment_size` | Bytes |
| **Packet Lengths (Bwd)** | `total_length_of_bwd_packets`, `bwd_packet_length_max`, `bwd_packet_length_min`, `bwd_packet_length_mean`, `bwd_packet_length_std`, `avg_bwd_segment_size` | Bytes |
| **Overall Lengths** | `min_packet_length`, `max_packet_length`, `packet_length_mean`, `packet_length_std`, `packet_length_variance`, `average_packet_size` | Bytes, variance in $\text{bytes}^2$ |
| **Packet Counts** | `total_fwd_packets`, `total_backward_packets`, `down_up_ratio`, `act_data_pkt_fwd` | Counts / ratio |
| **Inter-Arrival Times** | `flow_iat_mean`, `std`, `max`, `min`; `fwd_iat_total`, `mean`, `std`, `max`, `min`; `bwd_iat_total`, `mean`, `std`, `max`, `min` | Microseconds ($\mu s$) |
| **TCP Flags** | `fin_flag_count`, `syn_flag_count`, `rst_flag_count`, `psh_flag_count`, `ack_flag_count`, `urg_flag_count`, `cwe_flag_count`, `ece_flag_count`, `fwd_psh_flags`, `bwd_psh_flags`, `fwd_urg_flags`, `bwd_urg_flags` | Integer bitmask counts |
| **Headers & Windows** | `fwd_header_length`, `bwd_header_length`, `min_seg_size_forward`, `init_win_bytes_forward`, `init_win_bytes_backward` | Bytes |
| **Activity Bursts** | `active_mean`, `active_std`, `active_max`, `active_min`, `idle_mean`, `idle_std`, `idle_max`, `idle_min` | Microseconds ($\mu s$) |

---

## 3. Zeek vs. CICFlowMeter vs. Scapy Comparison

### What Standard Zeek (`conn.log`) Provides
In default out-of-the-box configurations, Zeek logs connection summaries:
* `ts`, `uid`, `id.orig_h`, `id.orig_p`, `id.resp_h`, `id.resp_p`, `proto`, `service`
* `duration`: connection duration in seconds (float)
* `orig_bytes`, `resp_bytes`: payload byte sums
* `conn_state`: connection state (e.g., `SF`, `S0`, `REJ`)
* `history`: state history string (e.g., `ShADadFf`)
* `orig_pkts`, `resp_pkts`: packet counts

### What Zeek Lacks for the Existing 66-Feature Model
1. **Variance & Standard Deviation of Packet Lengths:** Zeek logs total byte counts and total packets, but discards individual packet lengths after updating accumulators. Computing `fwd_packet_length_std` or `packet_length_variance` is impossible from default `conn.log`.
2. **Sub-Flow Inter-Arrival Times (IAT):** Zeek tracks start time and end time, but does not maintain timestamps of intermediate packets to compute IAT mean, standard deviation, max, or min.
3. **TCP Initial Window Sizes:** Standard `conn.log` does not record initial window advertisements (`init_win_bytes_forward`, `init_win_bytes_backward`).
4. **Active / Idle Microsecond Bursts:** Zeek does not segment flows into sub-bursts based on activity timeouts (5s idle gaps).
5. **Exact Flag Frequency Counts:** While Zeek's `history` string flags connection events, it does not tally total ACK, PSH, or URG flag counts.

### Comparative Matrix

| Capability / Metric | Zeek (`conn.log`) | CICFlowMeter (Java) | AI-NIDS Native Scapy Extractor |
|---|---|---|---|
| **Requires Paid License?** | No (Free/Open Source) | No (Free) | **No (Free / Open Source)** |
| **External Dependencies** | C++ compiler, libpcap | Java JRE/JDK | **Zero (Pure Python + Scapy)** |
| **Flow Duration** | Seconds | Microseconds ($\mu s$) | **Microseconds ($\mu s$)** |
| **Packet Length Std/Var** | ❌ No | ✅ Yes | **✅ Yes** |
| **Inter-Arrival Time (IAT) Stats** | ❌ No | ✅ Yes | **✅ Yes** |
| **TCP Flag Counts (SYN/ACK/etc.)** | ❌ (History string only) | ✅ Yes | **✅ Yes** |
| **Active/Idle Burst Stats** | ❌ No | ✅ Yes | **✅ Yes** |
| **Initial TCP Window Sizes** | ❌ No | ✅ Yes | **✅ Yes** |
| **Direct Model Compatibility** | ❌ Incompatible schema | ✅ Compatible | **✅ 100% Compatible (66 features)** |

---

## 4. Architectural Decision: Native Scapy Flow Extractor

Rather than forcing users to install external Java runtimes (CICFlowMeter) or write complex Zeek C++/event scripts (`event new_packet`), Phase 5 implements a native bi-directional flow accumulator in [`src/flow_features.py`](file:///Users/it4/Desktop/NetworkIntrusionDetection/src/flow_features.py) using Scapy:
* **Stream-Based Dissection:** Processes packets packet-by-packet using `PcapReader` to maintain $O(1)$ memory overhead even on multi-gigabyte PCAPs.
* **Exact Mathematical Alignment:** Matches CICFlowMeter's definitions for forward/backward packet partitioning, microsecond scaling, and burst idle thresholds.
* **Schema Validation Safety:** Extracted records are strictly validated against `models/model_metadata.json`.

---

## 5. Schema Discrepancy Policy: "Never Silently Substitute"

In accordance with strict safety requirements:
1. **No Silent Substitution:** If a user feeds a third-party log (e.g. Zeek `conn.log`, Suricata EVE JSON, or NetFlow) that does not match the 66-feature schema, AI-NIDS **does not silently guess, synthesize, or fill missing statistical features with zeroes**.
2. **Strict Validation:** The `validate_features(X, feature_names)` utility verifies all 66 features, raising an explicit `ValueError` if required features are absent.
3. **Alternative Workflow for Connection Logs:** If users desire to analyze Zeek `conn.log` files, a dedicated, separately trained connection-level model pipeline (trained specifically on Zeek features) must be utilized rather than corrupting flow-level models.

