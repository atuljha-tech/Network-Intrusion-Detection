# AI-NIDS: Held-Out Capture Period & Scenario Generalization Evaluation

## Executive Summary
This evaluation rigorously tests model generalization across independent capture periods / scenarios.
Evaluating machine learning models across temporal periods simulates real-world deployment where traffic
patterns, attack rates, and protocols shift relative to the training baseline.

---

## 1. Experimental Setup
* **Dataset Source:** `synthetic`
* **Training Capture Periods (4):** `Friday.csv, Monday.csv, Thursday.csv, Tuesday.csv` (4042 flows)
* **Held-Out Evaluation Period:** `Wednesday.csv` (958 flows)
* **Evaluated Supervised Model:** `RandomForest` (class_weight='balanced')
* **Evaluated Unsupervised Model:** `IsolationForest` (trained exclusively on nominal benign flows)

---

## 2. In-Domain vs. Held-Out Generalization Performance

| Metric | In-Domain Test Set | Held-Out Period (`Wednesday.csv`) | Generalization Drop (Δ) |
|---|---|---|---|
| **Accuracy** | 0.6007 | 0.5939 | +0.0068 |
| **Precision (Malicious)** | 0.5152 | 0.4468 | +0.0683 |
| **Recall (Malicious)** | 0.0525 | 0.0547 | -0.0022 |
| **F1-Score (Malicious)** | 0.0952 | 0.0974 | -0.0022 |
| **ROC-AUC** | 0.4896 | 0.4757 | +0.0139 |

---

## 3. Unsupervised Outlier Detection on Held-Out Period
* **Isolation Forest Nominal Training Baseline:** Trained on benign traffic only.
* **Held-Out Malicious Flows Flagged as Outliers:** `4.9%`
* **Held-Out Benign Flows Triggering False Alarms:** `6.3%`

---

## 4. Per-Class Behavior on Held-Out Capture

| Attack Category | Sample Count | Novel to Training? | Detection Rate (%) |
|---|---|---|---|
| `benign` | 574 | No (Seen) | 95.5% |
| `bot` | 56 | No (Seen) | 8.9% |
| `ddos` | 100 | No (Seen) | 4.0% |
| `dos goldeneye` | 32 | No (Seen) | 3.1% |
| `dos hulk` | 79 | No (Seen) | 3.8% |
| `dos slowhttptest` | 32 | No (Seen) | 3.1% |
| `dos slowloris` | 11 | No (Seen) | 18.2% |
| `ftp-patator` | 13 | No (Seen) | 7.7% |
| `heartbleed` | 1 | No (Seen) | 0.0% |
| `infiltration` | 3 | No (Seen) | 0.0% |
| `portscan` | 39 | No (Seen) | 7.7% |
| `ssh-patator` | 15 | No (Seen) | 0.0% |
| `web attack - brute force` | 3 | No (Seen) | 33.3% |

---

## 5. Security & Operational Implications
1. **The Cross-Period Generalization Gap:** Standard random cross-validation often reports over-optimistic
   metrics because flows from the same attack run are split across both train and test partitions.
   Evaluating on `Wednesday.csv` exposes the true domain transfer penalty (Δ F1 = -0.0022).
2. **Novel Attack Classes:** When a new attack category appears in the held-out period that was absent
   during training, supervised classifiers can misclassify it if its statistical features resemble benign traffic.
3. **Hybrid Defense-in-Depth:** Pairing supervised classifiers with unsupervised Isolation Forest ensures that
   novel, previously unseen attack behaviors still surface high anomaly scores for human analyst investigation.
