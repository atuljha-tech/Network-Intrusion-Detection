# AI-NIDS: Performance, Latency, Throughput & Memory Benchmark

## Executive Summary
Phase 7 performance profiling measures inference latency, batch throughput, packet dissection rate,
and memory consumption. All tests were executed locally on the host system without mock timers.

* **Host Platform:** `Darwin x86_64 (Python 3.11.9)`
* **Benchmark Execution Date:** `2026-10-25 17:44:32 UTC`

---

## 1. Single-Flow Inference Latency

Measured over 500 timed iterations per model (after 50 warmup iterations):

| Component | Median Latency | Mean Latency | 95th Percentile | 99th Percentile | Min / Max |
|---|---|---|---|---|---|
| **Binary Classifier** | `10.73 ms` | `13.69 ms` | `23.89 ms` | `63.20 ms` | `7.54 / 169.95 ms` |
| **Multiclass Classifier** | `10.98 ms` | `14.82 ms` | `33.01 ms` | `71.54 ms` | `7.49 / 91.97 ms` |
| **Anomaly Detection (Isolation Forest)** | `30.18 ms` | `35.79 ms` | `67.68 ms` | `99.28 ms` | `22.84 / 116.92 ms` |
| **Explainability (XAI Surrogate)** | `14.12 ms` | `21.56 ms` | `60.74 ms` | `104.83 ms` | `8.97 / 375.04 ms` |

---

## 2. Batch Inference Scaling & Throughput

Scaling behavior of the Multiclass pipeline across batch sizes (1 to 5,000 flows):

| Batch Size (Flows) | Total Latency (ms) | Latency per Flow (ms) | Throughput (Flows / Second) |
|---|---|---|---|
| `1` | `11.06 ms` | `11.059 ms` | **`90.4 flows/s`** |
| `10` | `17.97 ms` | `1.797 ms` | **`556.5 flows/s`** |
| `50` | `14.45 ms` | `0.289 ms` | **`3,459.2 flows/s`** |
| `100` | `15.07 ms` | `0.151 ms` | **`6,636.5 flows/s`** |
| `500` | `40.18 ms` | `0.080 ms` | **`12,443.6 flows/s`** |
| `1,000` | `45.51 ms` | `0.046 ms` | **`21,973.1 flows/s`** |
| `2,500` | `61.77 ms` | `0.025 ms` | **`40,475.8 flows/s`** |
| `5,000` | `130.48 ms` | `0.026 ms` | **`38,320.5 flows/s`** |

---

## 3. PCAP Ingestion & Flow Reconstruction Throughput

* **Test Capture File:** `sample_network_traffic.pcap`
* **Packets Processed:** `17` packets
* **Flows Reconstructed:** `7` bidirectional flows
* **Packet Dissection Throughput:** **`1,727.6 packets/s`**
* **End-to-End Processing (Dissection + ML + Anomaly):** **`378.3 packets/s`** (`155.8 flows/s`)

---

## 4. Memory Profiling & Footprint

* **Total Process Resident Set Size (RSS):** `218.58 MB`
* **Peak Tracemalloc Heap Allocation (5,000 flows):** `18.82 MB`
* **Active Tracemalloc Heap:** `2.54 MB`

---

## 5. Engineering Optimization Insights
1. **Vectorized Vector Transformations:** The scikit-learn preprocessing pipeline handles batch transformations
   with high efficiency, reaching several thousand flows/second at batch sizes ≥ 500.
2. **Single-Flow Streaming Overhead:** Single-flow prediction incurs per-call DataFrame construction overhead.
   For high-speed live network feeds, micro-batching (buffering 50–100 flows or flushing every 250ms)
   dramatically increases throughput by 10x–20x compared to 1-by-1 evaluation.
3. **Memory Guardrails:** Memory usage scales linearly with flow count; capping streaming PCAP readers at
   `MAX_PACKETS_STREAMING = 500,000` prevents OOM on commodity analyst laptops.
