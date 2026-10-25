"""
anomaly.py — Isolation Forest Anomaly Detection Experiment for AI-NIDS.

IMPORTANT ARCHITECTURAL & OPERATIONAL PRINCIPLE:
    Anomaly detection operates in an unsupervised or semi-supervised manner, identifying
    traffic flows that deviate statistically from baseline patterns.
    
    CRITICAL SAFETY CONSTRAINT:
    DO NOT TREAT ANOMALY SCORES AS CONFIRMED ATTACK LABELS.
    Statistical outliers can represent benign traffic spikes, novel legitimate applications,
    unusual transfer sizes, or protocol nuances. Conversely, low-and-slow stealthy attacks
    may blend into normal distributions. Anomaly scores must be used as triage signals
    for further investigation, NOT automated block rules or ground-truth attack categories.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import (
    ANOMALY_CONTAMINATION,
    ANOMALY_METADATA_PATH,
    ANOMALY_MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    setup_logging,
)
from src.model_utils import get_package_versions

logger = setup_logging()
log = logging.getLogger("ai_nids.anomaly")

ANOMALY_DISCLAIMER: str = (
    "DISCLAIMER: Anomaly detection identifies statistical outliers and behavioral deviations "
    "from training baselines. Anomaly scores and outlier flags DO NOT represent confirmed malicious "
    "attacks. Benign network anomalies (large backups, video streams, network reconfigurations) "
    "regularly trigger outlier flags, while stealthy, low-volume attacks may appear normal. "
    "Anomaly signals are purely diagnostic candidates for secondary triage and must not be treated "
    "as ground-truth attack classifications."
)


def build_isolation_forest_pipeline(
    contamination: float = ANOMALY_CONTAMINATION,
    n_estimators: int = 100,
    random_state: int = RANDOM_SEED,
    n_jobs: int = -1,
) -> Pipeline:
    """Create an Isolation Forest pipeline with standard median imputation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("isolation_forest", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])


def train_isolation_forest(
    X_train: pd.DataFrame,
    y_train_binary: Optional[pd.Series] = None,
    fit_on_benign_only: bool = True,
    contamination: float = ANOMALY_CONTAMINATION,
    n_estimators: int = 100,
) -> Tuple[Pipeline, Dict]:
    """Train the Isolation Forest anomaly detector.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train_binary : pd.Series | None
        Binary labels (0=Benign, 1=Malicious). If provided and ``fit_on_benign_only`` is True,
        the model fits solely on benign network flows to model nominal behavior.
    fit_on_benign_only : bool
        Whether to restrict training data to benign baseline traffic.
    contamination : float
        Expected fraction of outliers in the data.
    n_estimators : int
        Number of isolation trees.

    Returns
    -------
    (pipeline, training_info_dict)
    """
    pipeline = build_isolation_forest_pipeline(
        contamination=contamination,
        n_estimators=n_estimators,
    )

    if fit_on_benign_only and y_train_binary is not None:
        benign_mask = y_train_binary == 0
        X_fit = X_train[benign_mask].copy()
        training_mode = "semi_supervised_benign_baseline"
        log.info(
            "Fitting Isolation Forest on %d benign baseline flows (unsupervised nominal profile)...",
            len(X_fit),
        )
    else:
        X_fit = X_train.copy()
        training_mode = "unsupervised_general_traffic"
        log.info("Fitting Isolation Forest on %d unlabelled traffic flows...", len(X_fit))

    start = time.time()
    pipeline.fit(X_fit)
    elapsed = time.time() - start
    log.info("Isolation Forest fitted in %.2f s", elapsed)

    info = {
        "training_mode": training_mode,
        "training_samples": len(X_fit),
        "contamination": contamination,
        "n_estimators": n_estimators,
        "fit_time_seconds": elapsed,
        "disclaimer": ANOMALY_DISCLAIMER,
    }
    return pipeline, info


def compute_anomaly_scores(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute raw, decision, and calibrated anomaly scores.

    In scikit-learn's Isolation Forest:
      • ``decision_function(X)``: < 0 denotes anomaly, > 0 denotes normal.
      • ``score_samples(X)``: opposite of anomaly score (lower is more anomalous).

    Returns
    -------
    (calibrated_scores, raw_scores, is_outlier_flag)
      • ``calibrated_scores``: Float array in [0.0, 1.0] where 1.0 = highly anomalous,
        0.0 = highly normal.
      • ``raw_scores``: Raw output of ``score_samples(X)``.
      • ``is_outlier_flag``: Boolean array (True if flagged as statistical outlier).
    """
    raw_scores = pipeline.score_samples(X)
    decision = pipeline.decision_function(X)
    is_outlier = pipeline.predict(X) == -1

    # Calibrate scores to [0, 1] using logistic sigmoid on inverted decision function
    # decision > 0 -> normal -> score < 0.5
    # decision < 0 -> anomalous -> score > 0.5
    calibrated_scores = 1.0 / (1.0 + np.exp(10.0 * decision))

    return calibrated_scores, raw_scores, is_outlier


def evaluate_anomaly_detection(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_binary: pd.Series,
    y_test_multiclass: Optional[pd.Series] = None,
) -> Dict:
    """Evaluate Isolation Forest anomaly scores against traffic categories.

    Note: The anomaly detector was trained WITHOUT using attack labels.
    Evaluation compares how the unsupervised anomaly scores differentiate
    benign from malicious traffic, emphasizing that anomalies are NOT ground-truth labels.
    """
    calibrated_scores, raw_scores, is_outlier = compute_anomaly_scores(pipeline, X_test)

    # ROC-AUC using calibrated anomaly scores vs binary target
    try:
        auc_roc = float(roc_auc_score(y_test_binary, calibrated_scores))
    except Exception as exc:
        log.warning("Could not compute anomaly ROC-AUC: %s", exc)
        auc_roc = None

    try:
        auc_pr = float(average_precision_score(y_test_binary, calibrated_scores))
    except Exception as exc:
        log.warning("Could not compute anomaly PR-AUC: %s", exc)
        auc_pr = None

    # Overall outlier rates
    total_test = len(X_test)
    outlier_count = int(np.sum(is_outlier))
    outlier_pct = float((outlier_count / total_test) * 100.0)

    # False alarm rate on Benign traffic
    benign_mask = y_test_binary == 0
    benign_total = int(np.sum(benign_mask))
    benign_flagged = int(np.sum(is_outlier[benign_mask])) if benign_total > 0 else 0
    benign_false_alarm_rate = (
        float((benign_flagged / benign_total) * 100.0) if benign_total > 0 else 0.0
    )

    # Malicious detection rate (unsupervised recall)
    malicious_mask = y_test_binary == 1
    malicious_total = int(np.sum(malicious_mask))
    malicious_flagged = int(np.sum(is_outlier[malicious_mask])) if malicious_total > 0 else 0
    malicious_outlier_rate = (
        float((malicious_flagged / malicious_total) * 100.0) if malicious_total > 0 else 0.0
    )

    # Per-attack category outlier analysis
    per_class_outlier_rates: Dict[str, Dict] = {}
    if y_test_multiclass is not None:
        for cname in y_test_multiclass.unique():
            c_mask = y_test_multiclass == cname
            c_total = int(np.sum(c_mask))
            c_flagged = int(np.sum(is_outlier[c_mask]))
            c_scores = calibrated_scores[c_mask]
            per_class_outlier_rates[str(cname)] = {
                "support": c_total,
                "outliers_flagged": c_flagged,
                "outlier_rate_pct": round((c_flagged / max(1, c_total)) * 100.0, 2),
                "mean_anomaly_score": round(float(np.mean(c_scores)), 4),
            }

    results: Dict = {
        "disclaimer": ANOMALY_DISCLAIMER,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "total_test_samples": total_test,
        "total_outliers_flagged": outlier_count,
        "overall_outlier_pct": round(outlier_pct, 2),
        "benign_samples": benign_total,
        "benign_false_alarm_count": benign_flagged,
        "benign_false_alarm_rate_pct": round(benign_false_alarm_rate, 2),
        "malicious_samples": malicious_total,
        "malicious_flagged_count": malicious_flagged,
        "malicious_outlier_rate_pct": round(malicious_outlier_rate, 2),
        "per_class_outlier_analysis": per_class_outlier_rates,
    }

    log.info(
        "Isolation Forest Evaluation — AUC-ROC: %s | Malicious Flagged: %.1f%% | Benign False Alarm: %.1f%%",
        f"{auc_roc:.4f}" if auc_roc is not None else "N/A",
        malicious_outlier_rate,
        benign_false_alarm_rate,
    )
    return results


def plot_anomaly_distribution(
    calibrated_scores: np.ndarray,
    y_test_binary: pd.Series,
    output_path: Optional[Path] = None,
) -> Path:
    """Plot anomaly score distributions for benign vs malicious traffic."""
    output_path = output_path or (REPORTS_DIR / "anomaly_score_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    benign_scores = calibrated_scores[y_test_binary == 0]
    malicious_scores = calibrated_scores[y_test_binary == 1]

    ax.hist(
        benign_scores,
        bins=30,
        alpha=0.6,
        label=f"Benign Flows (N={len(benign_scores)})",
        color="#2b5c8f",
        density=True,
    )
    ax.hist(
        malicious_scores,
        bins=30,
        alpha=0.6,
        label=f"Malicious Flows (N={len(malicious_scores)})",
        color="#c83838",
        density=True,
    )

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")
    ax.set_title("Isolation Forest Calibrated Anomaly Score Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Calibrated Anomaly Score [0 = Normal, 1 = Outlier]")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved anomaly score distribution plot → %s", output_path)
    return output_path


def format_anomaly_report(results: Dict, info: Dict) -> str:
    """Format anomaly experiment results into a readable report."""
    lines = [
        "=" * 70,
        "  AI-NIDS Isolation Forest Anomaly Detection Experiment Report",
        "=" * 70,
        "CRITICAL NOTICE / DISCLAIMER:",
        f"  {results.get('disclaimer', ANOMALY_DISCLAIMER)}",
        "-" * 70,
        "Experimental Setup:",
        f"  Training Mode:         {info.get('training_mode')}",
        f"  Training Samples:      {info.get('training_samples'):,}",
        f"  Contamination Param:   {info.get('contamination')}",
        f"  Number of Trees:       {info.get('n_estimators')}",
        f"  Fit Time:              {info.get('fit_time_seconds', 0):.2f} s",
        "-" * 70,
        "Unsupervised Separation Performance:",
        f"  ROC-AUC vs Ground Truth:  {results.get('auc_roc', 'N/A')}",
        f"  PR-AUC vs Ground Truth:   {results.get('auc_pr', 'N/A')}",
        f"  Overall Outlier Rate:     {results.get('overall_outlier_pct', 0):.2f}% ({results.get('total_outliers_flagged', 0):,} / {results.get('total_test_samples', 0):,})",
        f"  Benign False Alarm Rate:  {results.get('benign_false_alarm_rate_pct', 0):.2f}% ({results.get('benign_false_alarm_count', 0):,} / {results.get('benign_samples', 0):,})",
        f"  Malicious Outlier Rate:   {results.get('malicious_outlier_rate_pct', 0):.2f}% ({results.get('malicious_flagged_count', 0):,} / {results.get('malicious_samples', 0):,})",
        "-" * 70,
        "Per-Attack Category Outlier Breakdown:",
        f"  {'Category':<28} {'Total':>8} {'Flagged':>8} {'Rate':>8} {'Mean Score':>12}",
        f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*12}",
    ]

    for cname, stats in results.get("per_class_outlier_analysis", {}).items():
        lines.append(
            f"  {cname:<28} {stats['support']:>8,} {stats['outliers_flagged']:>8,} {stats['outlier_rate_pct']:>7.1f}% {stats['mean_anomaly_score']:>12.4f}"
        )

    lines.extend([
        "-" * 70,
        "Operational Guidance:",
        "  1. Anomaly detection flags statistical anomalies, which frequently include benign novelties.",
        "  2. Do not automate punitive firewall blocks based purely on anomaly scores.",
        "  3. Use anomaly scores as priority multipliers during security analyst triage.",
        "=" * 70,
    ])
    return "\n".join(lines)


def run_anomaly_experiment(
    X_train: pd.DataFrame,
    y_train_binary: pd.Series,
    X_test: pd.DataFrame,
    y_test_binary: pd.Series,
    y_test_multiclass: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    save_artifacts: bool = True,
) -> Tuple[Pipeline, Dict, Dict]:
    """Execute end-to-end Isolation Forest anomaly detection experiment.

    Returns ``(pipeline, info_dict, results_dict)``.
    """
    pipeline, info = train_isolation_forest(
        X_train,
        y_train_binary=y_train_binary,
        fit_on_benign_only=True,
    )

    results = evaluate_anomaly_detection(
        pipeline,
        X_test,
        y_test_binary,
        y_test_multiclass=y_test_multiclass,
    )

    if save_artifacts:
        import joblib
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, ANOMALY_MODEL_PATH)
        log.info("Saved Isolation Forest model → %s", ANOMALY_MODEL_PATH)

        calibrated_scores, _, _ = compute_anomaly_scores(pipeline, X_test)
        plot_anomaly_distribution(calibrated_scores, y_test_binary)

        metadata = {
            "model_type": "IsolationForest",
            "training_date": pd.Timestamp.now().isoformat(),
            "feature_names": feature_names or list(X_train.columns),
            "n_features": len(feature_names or list(X_train.columns)),
            "info": info,
            "evaluation": results,
            "package_versions": get_package_versions(),
            "disclaimer": ANOMALY_DISCLAIMER,
        }
        with open(ANOMALY_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        log.info("Saved anomaly metadata → %s", ANOMALY_METADATA_PATH)

        report_text = format_anomaly_report(results, info)
        txt_path = REPORTS_DIR / "anomaly_detection_report.txt"
        txt_path.write_text(report_text, encoding="utf-8")
        log.info("Saved anomaly report → %s", txt_path)

    return pipeline, info, results

