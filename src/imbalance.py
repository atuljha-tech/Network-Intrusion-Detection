"""
imbalance.py — Class Imbalance Analysis for AI-NIDS.

Analyzes class distributions, computes imbalance ratios, evaluates the disparity
between macro and weighted metrics, and assesses the impact of class weighting on
minority attack detection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, setup_logging

logger = setup_logging()
log = logging.getLogger("ai_nids.imbalance")


def compute_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Compute distribution statistics for a class series.

    Returns a DataFrame with columns:
        * ``class_name``
        * ``count``
        * ``percentage``
        * ``imbalance_ratio_vs_max`` (majority_count / class_count)
    """
    counts = y.value_counts()
    total = len(y)
    max_count = counts.max()

    df = pd.DataFrame({
        "class_name": counts.index.astype(str),
        "count": counts.values,
        "percentage": (counts.values / total) * 100.0,
        "imbalance_ratio_vs_max": [round(max_count / max(1, c), 2) for c in counts.values],
    })
    return df


def calculate_gini_imbalance(counts: np.ndarray) -> float:
    """Compute the Gini coefficient of class size inequality.

    A value of 0 indicates perfect class equality; values close to 1
    indicate extreme class imbalance.
    """
    if len(counts) <= 1:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(counts)
    cum_counts = np.cumsum(sorted_counts)
    gini = (2.0 * np.sum((np.arange(1, n + 1) * sorted_counts)) - (n + 1) * cum_counts[-1]) / (n * cum_counts[-1])
    return float(max(0.0, min(1.0, gini)))


def analyze_class_imbalance(
    y_train: pd.Series,
    y_test: Optional[pd.Series] = None,
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    macro_f1: Optional[float] = None,
    weighted_f1: Optional[float] = None,
    model_name: str = "Multiclass Classifier",
) -> Dict:
    """Analyze class imbalance in training and test data and its impact on metrics.

    Parameters
    ----------
    y_train : pd.Series
        Training class labels.
    y_test : pd.Series | None
        Held-out test class labels (optional).
    per_class_metrics : dict | None
        Dictionary mapping class_name -> {precision, recall, f1, support}.
    macro_f1 : float | None
        Overall macro F1 score.
    weighted_f1 : float | None
        Overall weighted F1 score.
    model_name : str
        Name of model evaluated.

    Returns
    -------
    dict
        Comprehensive imbalance analysis dictionary.
    """
    train_dist_df = compute_class_distribution(y_train)
    total_samples = len(y_train)
    counts = train_dist_df["count"].values

    majority_class = train_dist_df.iloc[0]["class_name"]
    majority_count = int(train_dist_df.iloc[0]["count"])
    minority_class = train_dist_df.iloc[-1]["class_name"]
    minority_count = int(train_dist_df.iloc[-1]["count"])
    overall_imbalance_ratio = float(majority_count / max(1, minority_count))
    gini_coeff = calculate_gini_imbalance(counts)

    # Identify classes with <= 1% representation (extreme minority)
    extreme_minority = train_dist_df[train_dist_df["percentage"] <= 1.0]["class_name"].tolist()
    moderate_minority = train_dist_df[
        (train_dist_df["percentage"] > 1.0) & (train_dist_df["percentage"] < 5.0)
    ]["class_name"].tolist()

    analysis: Dict = {
        "model_name": model_name,
        "total_training_samples": total_samples,
        "n_classes": len(train_dist_df),
        "majority_class": majority_class,
        "majority_percentage": float(train_dist_df.iloc[0]["percentage"]),
        "minority_class": minority_class,
        "minority_percentage": float(train_dist_df.iloc[-1]["percentage"]),
        "imbalance_ratio": overall_imbalance_ratio,
        "gini_coefficient": round(gini_coeff, 4),
        "extreme_minority_classes_lt_1pct": extreme_minority,
        "moderate_minority_classes_1_to_5pct": moderate_minority,
        "class_distribution": train_dist_df.to_dict(orient="records"),
    }

    # Evaluate metric disparity if metrics provided
    if macro_f1 is not None and weighted_f1 is not None:
        disparity = float(weighted_f1 - macro_f1)
        analysis["macro_f1"] = macro_f1
        analysis["weighted_f1"] = weighted_f1
        analysis["metric_disparity_weighted_minus_macro"] = round(disparity, 4)
        analysis["imbalance_masking_effect"] = (
            "Severe: Weighted F1 is significantly higher than Macro F1, "
            "indicating that majority-class success is concealing low performance on rare attacks."
            if disparity > 0.08
            else "Moderate/Low: Macro and weighted F1 are relatively aligned."
        )

    # Analyze correlation between support and recall
    if per_class_metrics:
        supports = []
        recalls = []
        f1s = []
        for cname, pcm in per_class_metrics.items():
            supports.append(pcm.get("support", 0))
            recalls.append(pcm.get("recall", 0.0))
            f1s.append(pcm.get("f1", 0.0))

        if len(supports) > 2 and np.std(supports) > 0 and np.std(recalls) > 0:
            corr_recall = float(np.corrcoef(supports, recalls)[0, 1])
            corr_f1 = float(np.corrcoef(supports, f1s)[0, 1])
        else:
            corr_recall = 0.0
            corr_f1 = 0.0

        analysis["support_recall_correlation"] = round(corr_recall, 4)
        analysis["support_f1_correlation"] = round(corr_f1, 4)
        analysis["correlation_interpretation"] = (
            "Positive correlation between class support and recall confirms that attacks with fewer "
            "training samples suffer poorer detection rates under standard loss formulations."
            if corr_recall > 0.3
            else "Class support has moderate or low correlation with recall under current weighting."
        )

    # Mitigation recommendations
    analysis["mitigations"] = [
        "Incorporate 'balanced' class weights (inverse frequency weighting) in loss calculation.",
        "Evaluate primarily on Macro F1 and per-class recall rather than weighted metrics or overall accuracy.",
        "Establish specialized alert routing or threshold tuning for high-risk rare attacks.",
        "Consider focused synthetic oversampling (SMOTE) or targeted network capture augmentation.",
    ]

    return analysis


def format_imbalance_report(analysis: Dict) -> str:
    """Format the imbalance analysis into a detailed text report."""
    lines = [
        "=" * 70,
        f"  AI-NIDS Class Imbalance Analysis Report — {analysis.get('model_name', 'Model')}",
        "=" * 70,
        f"Total Samples:       {analysis.get('total_training_samples', 0):,}",
        f"Unique Classes:      {analysis.get('n_classes', 0)}",
        f"Majority Class:      {analysis.get('majority_class')} ({analysis.get('majority_percentage', 0):.2f}%)",
        f"Minority Class:      {analysis.get('minority_class')} ({analysis.get('minority_percentage', 0):.2f}%)",
        f"Imbalance Ratio:     {analysis.get('imbalance_ratio', 0):.1f} : 1",
        f"Gini Inequality:     {analysis.get('gini_coefficient', 0):.4f} (0 = equal, 1 = extreme disparity)",
        "-" * 70,
        "Class Distribution Breakdown:",
        f"  {'Class Name':<28} {'Count':>8} {'Percent':>8} {'Ratio vs Max':>14}",
        f"  {'-'*28} {'-'*8} {'-'*8} {'-'*14}",
    ]

    for item in analysis.get("class_distribution", []):
        lines.append(
            f"  {item['class_name']:<28} {item['count']:>8,} {item['percentage']:>7.2f}% {item['imbalance_ratio_vs_max']:>13.1f}x"
        )

    lines.extend([
        "-" * 70,
        "Critical Observations on Metric Disparity:",
    ])

    if "macro_f1" in analysis:
        lines.extend([
            f"  Macro F1:          {analysis['macro_f1']:.4f}",
            f"  Weighted F1:       {analysis['weighted_f1']:.4f}",
            f"  Disparity (W - M): {analysis.get('metric_disparity_weighted_minus_macro', 0):.4f}",
            f"  Assessment:        {analysis.get('imbalance_masking_effect', '')}",
        ])

    if "support_recall_correlation" in analysis:
        lines.extend([
            f"  Support-Recall r:  {analysis.get('support_recall_correlation', 0):.4f}",
            f"  Support-F1 r:      {analysis.get('support_f1_correlation', 0):.4f}",
            f"  Assessment:        {analysis.get('correlation_interpretation', '')}",
        ])

    lines.extend([
        "-" * 70,
        "Imbalance Mitigation Strategies:",
    ])
    for mit in analysis.get("mitigations", []):
        lines.append(f"  • {mit}")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_imbalance_analysis(
    analysis: Dict,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Save imbalance analysis as both JSON and TXT report."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "class_imbalance_analysis.json"
    txt_path = output_dir / "class_imbalance_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)

    report_text = format_imbalance_report(analysis)
    txt_path.write_text(report_text, encoding="utf-8")

    log.info("Saved imbalance JSON report → %s", json_path)
    log.info("Saved imbalance text report → %s", txt_path)
    return json_path, txt_path
