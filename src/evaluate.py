"""
evaluate.py — Model evaluation and comparison for AI-NIDS Phase 3.

Computes accuracy, precision, recall, F1, confusion matrix, ROC-AUC,
and timing for each trained model.  Generates plots and comparison
reports.

Terminology for binary intrusion detection (positive class = 1 = Malicious):
  • True  Positive (TP): malicious flow correctly detected as malicious.
  • False Positive (FP): benign flow incorrectly flagged as malicious.
  • False Negative (FN): malicious flow missed (classified as benign).
  • True  Negative (TN): benign flow correctly classified as benign.

For a security system, malicious-class recall (FN minimisation) and
precision (FP minimisation) deserve particular attention.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MIN_RECALL_THRESHOLD, REPORTS_DIR, setup_logging

logger = setup_logging()
log = logging.getLogger("ai_nids.evaluate")


# ── single-model evaluation ─────────────────────────────────────────────────

def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> Dict:
    """Evaluate a fitted model pipeline on the held-out test set.

    Returns a dict containing all metrics, the confusion matrix,
    and a formatted classification report.
    """
    # Predict and time it
    start = time.time()
    y_pred = pipeline.predict(X_test)
    predict_time = time.time() - start

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Benign", "Malicious"],
            zero_division=0,
        ),
        "predict_time": float(predict_time),
    }

    # ROC-AUC — requires probability or decision-function scores
    roc_auc = _safe_roc_auc(pipeline, X_test, y_test, model_name)
    metrics["roc_auc"] = roc_auc

    # Confusion-matrix decomposition
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    })

    # Check recall against configurable threshold
    metrics["meets_recall_threshold"] = metrics["recall"] >= MIN_RECALL_THRESHOLD

    log.info(
        "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f%s%s",
        model_name,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
        f" | AUC: {roc_auc:.4f}" if roc_auc is not None else "",
        "" if metrics["meets_recall_threshold"]
        else f" ⚠ recall < {MIN_RECALL_THRESHOLD:.0%}",
    )
    return metrics


def _safe_roc_auc(pipeline, X_test, y_test, model_name: str):
    """Compute ROC-AUC, returning None if not possible."""
    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            return float(roc_auc_score(y_test, y_proba))
        if hasattr(pipeline, "decision_function"):
            y_scores = pipeline.decision_function(X_test)
            return float(roc_auc_score(y_test, y_scores))
        log.warning(
            "%s does not support probability scores — ROC-AUC skipped.",
            model_name,
        )
    except ValueError as exc:
        log.warning("ROC-AUC failed for %s: %s", model_name, exc)
    return None


# ── model comparison and selection ──────────────────────────────────────────

def compare_models(results: Dict[str, Dict]) -> Tuple[str, pd.DataFrame]:
    """Compare all trained models and select the best one.

    Selection criteria
    ------------------
    1. Among models meeting ``MIN_RECALL_THRESHOLD``, select the one
       with the highest malicious-class F1-score.
    2. If no model meets the threshold, select the one with the
       highest recall (to prioritise detection).

    Returns ``(best_model_name, comparison_dataframe)``.
    """
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
            "train_time": result["train_time"],
            "predict_time": m["predict_time"],
            "meets_recall_threshold": m["meets_recall_threshold"],
        })

    comparison = pd.DataFrame(rows)

    # Selection logic
    eligible = comparison[comparison["meets_recall_threshold"]]

    if not eligible.empty:
        best_idx = eligible["f1"].idxmax()
        best_name = str(eligible.loc[best_idx, "model"])
        log.info(
            "Selected '%s' — highest F1 (%.4f) among models meeting "
            "recall threshold (≥ %.0f%%).",
            best_name, eligible.loc[best_idx, "f1"],
            MIN_RECALL_THRESHOLD * 100,
        )
    else:
        log.warning(
            "No model meets the minimum recall threshold of %.0f%%. "
            "Selecting model with highest recall.",
            MIN_RECALL_THRESHOLD * 100,
        )
        best_idx = comparison["recall"].idxmax()
        best_name = str(comparison.loc[best_idx, "model"])

    # Annotate highest-accuracy model if different
    highest_acc_idx = comparison["accuracy"].idxmax()
    highest_acc_name = str(comparison.loc[highest_acc_idx, "model"])
    if highest_acc_name != best_name:
        log.info(
            "Note: '%s' has the highest accuracy (%.4f) but '%s' was "
            "selected for better malicious-class detection.",
            highest_acc_name,
            comparison.loc[highest_acc_idx, "accuracy"],
            best_name,
        )

    return best_name, comparison


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save a confusion-matrix heatmap as PNG."""
    output_dir = output_dir or (REPORTS_DIR / "confusion_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Benign (0)", "Malicious (1)"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted label", ylabel="True label",
        title=f"Confusion Matrix — {model_name}",
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = output_dir / f"cm_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", path)
    return path


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a grouped bar chart comparing model metrics."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    if "roc_auc" in comparison_df.columns and comparison_df["roc_auc"].notna().all():
        metrics_to_plot.append("roc_auc")

    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, metric in enumerate(metrics_to_plot):
        values = comparison_df[metric].astype(float).values
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=metric.replace("_", " ").title(),
            color=colours[i % len(colours)],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.axhline(
        y=MIN_RECALL_THRESHOLD, color="red", linestyle="--", alpha=0.5,
        label=f"Min recall ({MIN_RECALL_THRESHOLD:.0%})",
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(
        "Model Comparison — Binary Intrusion Detection",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved model comparison chart → %s", path)
    return path


# ── report persistence ──────────────────────────────────────────────────────

def save_training_summary(
    results: Dict[str, Dict],
    best_name: str,
    metadata: Dict,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save a comprehensive training summary as JSON."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {
        "selected_model": best_name,
        "training_date": metadata.get("training_date"),
        "data_source": metadata.get("data_source"),
        "train_samples": metadata.get("train_samples"),
        "test_samples": metadata.get("test_samples"),
        "n_features": metadata.get("n_features"),
        "min_recall_threshold": MIN_RECALL_THRESHOLD,
        "models": {},
    }

    for name, result in results.items():
        m = result["metrics"]
        cm = m["confusion_matrix"]
        summary["models"][name] = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
            "train_time": result["train_time"],
            "predict_time": m["predict_time"],
            "meets_recall_threshold": m["meets_recall_threshold"],
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        }

    path = output_dir / "training_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Training summary saved → %s", path)
    return path


def save_classification_reports(
    results: Dict[str, Dict],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Save per-model classification reports as text files."""
    output_dir = output_dir or (REPORTS_DIR / "classification_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for name, result in results.items():
        report_text = result["metrics"]["classification_report"]
        safe_name = name.lower().replace(" ", "_")
        path = output_dir / f"report_{safe_name}.txt"
        path.write_text(report_text, encoding="utf-8")
        paths.append(path)

    log.info("Saved %d classification report(s) to %s", len(paths), output_dir)
    return paths


# ── Multiclass Evaluation (Phase 4) ──────────────────────────────────────────

def evaluate_multiclass_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: Optional[List[str]] = None,
    model_name: str = "multiclass_model",
) -> Dict:
    """Evaluate a fitted multiclass model pipeline on the held-out test set.

    Computes overall accuracy, macro F1, weighted F1, per-class precision,
    per-class recall, per-class F1, support, and full confusion matrix.
    """
    start = time.time()
    y_pred = pipeline.predict(X_test)
    predict_time = time.time() - start

    if class_names is None:
        if hasattr(pipeline, "classes_"):
            class_names = [str(c) for c in pipeline.classes_]
        elif (
            hasattr(pipeline, "named_steps")
            and hasattr(pipeline.named_steps.get("classifier", None), "classes_")
        ):
            class_names = [str(c) for c in pipeline.named_steps["classifier"].classes_]
        else:
            class_names = sorted(
                list(set(y_test.astype(str).unique()) | set(pd.Series(y_pred).astype(str).unique()))
            )

    class_names_str = [str(c) for c in class_names]
    y_test_str = y_test.astype(str)
    y_pred_str = pd.Series(y_pred, index=y_test.index).astype(str)

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    precisions = precision_score(
        y_test_str, y_pred_str, average=None, labels=class_names_str, zero_division=0
    )
    recalls = recall_score(
        y_test_str, y_pred_str, average=None, labels=class_names_str, zero_division=0
    )
    f1s = f1_score(
        y_test_str, y_pred_str, average=None, labels=class_names_str, zero_division=0
    )

    test_counts = y_test_str.value_counts().to_dict()
    for idx, c in enumerate(class_names_str):
        per_class[c] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx]),
            "support": int(test_counts.get(c, 0)),
        }

    cm = confusion_matrix(y_test_str, y_pred_str, labels=class_names_str)

    clf_report = classification_report(
        y_test_str,
        y_pred_str,
        labels=class_names_str,
        target_names=class_names_str,
        zero_division=0,
    )

    macro_f1 = float(f1_score(y_test_str, y_pred_str, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_test_str, y_pred_str, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_test_str, y_pred_str, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_test_str, y_pred_str))

    metrics: Dict = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "class_names": class_names_str,
        "classification_report": clf_report,
        "predict_time": float(predict_time),
    }

    # OvR ROC-AUC if predict_proba available
    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)
            if y_proba.shape[1] == len(class_names_str) and len(set(y_test_str)) > 1:
                metrics["macro_roc_auc"] = float(
                    roc_auc_score(
                        y_test_str,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=class_names_str,
                    )
                )
            else:
                metrics["macro_roc_auc"] = None
        else:
            metrics["macro_roc_auc"] = None
    except Exception as exc:
        log.warning("Could not compute multiclass ROC-AUC for %s: %s", model_name, exc)
        metrics["macro_roc_auc"] = None

    log.info(
        "Multiclass %s — Acc: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
        model_name,
        accuracy,
        macro_f1,
        weighted_f1,
    )
    return metrics


def compare_multiclass_models(results: Dict[str, Dict]) -> Tuple[str, pd.DataFrame]:
    """Compare all trained multiclass models and select the best one.

    Selection prioritises macro F1 because macro F1 gives equal weight to rare
    attack categories rather than allowing dominant benign traffic to mask
    detection failures on critical minority attacks.

    Returns ``(best_model_name, comparison_dataframe)``.
    """
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "macro_precision": m["macro_precision"],
            "weighted_precision": m["weighted_precision"],
            "macro_recall": m["macro_recall"],
            "weighted_recall": m["weighted_recall"],
            "macro_roc_auc": m.get("macro_roc_auc"),
            "train_time": result["train_time"],
            "predict_time": m["predict_time"],
        })

    comparison = pd.DataFrame(rows)
    best_idx = comparison["macro_f1"].idxmax()
    best_name = str(comparison.loc[best_idx, "model"])
    log.info("Selected best multiclass model: '%s' (Macro F1 = %.4f)", best_name, comparison.loc[best_idx, "macro_f1"])
    return best_name, comparison


def plot_multiclass_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save a multiclass confusion matrix heatmap as PNG."""
    output_dir = output_dir or (REPORTS_DIR / "confusion_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    figsize = (max(8.0, n_classes * 0.75), max(6.0, n_classes * 0.6))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=f"Multiclass Confusion Matrix — {model_name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2 if cm.max() > 0 else 1
    if n_classes <= 15:
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                ax.text(
                    j, i, f"{val:,}" if val > 0 else "0",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8 if n_classes > 8 else 10,
                )

    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = output_dir / f"cm_multiclass_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved multiclass confusion matrix → %s", path)
    return path


def save_multiclass_reports(
    results: Dict[str, Dict],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Save multiclass classification reports and per-class metrics CSV."""
    output_dir = output_dir or (REPORTS_DIR / "classification_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}
    all_per_class_rows = []

    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_")
        report_text = result["metrics"]["classification_report"]
        report_path = output_dir / f"multiclass_report_{safe_name}.txt"
        report_path.write_text(report_text, encoding="utf-8")
        saved_paths[f"report_{name}"] = report_path

        # Gather per-class metrics
        for cname, pcm in result["metrics"]["per_class_metrics"].items():
            all_per_class_rows.append({
                "model": name,
                "class_name": cname,
                "precision": pcm["precision"],
                "recall": pcm["recall"],
                "f1": pcm["f1"],
                "support": pcm["support"],
            })

    if all_per_class_rows:
        per_class_df = pd.DataFrame(all_per_class_rows)
        csv_path = REPORTS_DIR / "multiclass_per_class_metrics.csv"
        per_class_df.to_csv(csv_path, index=False)
        saved_paths["per_class_metrics_csv"] = csv_path
        log.info("Saved per-class metrics CSV → %s", csv_path)

    return saved_paths


# ── Held-out Capture Period & Generalization Evaluation (Phase 7 Task 8) ─────

def evaluate_held_out_generalization(
    df: Optional[pd.DataFrame] = None,
    source_col: Optional[str] = None,
    held_out_source: Optional[str] = None,
    output_dir: Optional[Path] = None,
    force_synthetic: bool = False,
) -> Dict[str, Any]:
    """Evaluate generalization using a held-out temporal capture period or attack scenario.

    Parameters
    ----------
    df : pd.DataFrame | None
        Input dataset. If None, resolved from processed CSV or synthetic generator.
    source_col : str | None
        Column designating the capture source or period (defaults to SOURCE_FILE_COLUMN).
    held_out_source : str | None
        Name of the source period to hold out for evaluation (defaults to 'Friday.csv').
    output_dir : Path | None
        Directory where generalization report is saved.
    force_synthetic : bool
        Whether to force synthetic data generation.

    Returns
    -------
    dict containing in-domain vs held-out comparison metrics, per-class breakdown,
    and path to the generated report.
    """
    from src.config import (
        BINARY_TARGET_COLUMN,
        ORIGINAL_LABEL_COLUMN,
        REPORTS_DIR,
        SOURCE_FILE_COLUMN,
    )
    from src.feature_engineering import separate_features_target
    from src.train import (
        _resolve_data,
        build_model_pipelines,
        prepare_features,
        split_data,
        train_single_model,
    )
    from src.anomaly import (
        build_isolation_forest_pipeline,
        compute_anomaly_scores,
    )

    out_dir = output_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    source_col = source_col or SOURCE_FILE_COLUMN

    # 1. Resolve Data
    if df is None:
        df, data_source = _resolve_data(None, force_synthetic=force_synthetic)
    else:
        data_source = "provided"

    log.info("Running Held-out Generalization Evaluation (data source: %s)...", data_source)

    # 2. Identify Sources / Capture Periods
    if source_col not in df.columns or df[source_col].nunique() <= 1:
        # Fallback scenario: partition by original label or temporal order
        log.warning("Source column '%s' not found or has single value. Partitioning by attack scenario.", source_col)
        unique_sources = ["Baseline_Training_Set", "Held_Out_Scenario"]
        # Split 75% baseline vs 25% held-out
        cutoff = int(len(df) * 0.75)
        df = df.copy()
        df[source_col] = ["Baseline_Training_Set"] * cutoff + ["Held_Out_Scenario"] * (len(df) - cutoff)
        held_out_source = "Held_Out_Scenario"
    else:
        unique_sources = sorted(list(df[source_col].unique()))
        if held_out_source is None or held_out_source not in unique_sources:
            # Pick the last source period (e.g. 'Friday.csv')
            held_out_source = unique_sources[-1]

    training_sources = [s for s in unique_sources if s != held_out_source]
    log.info("Generalization Setup — Training Sources: %s | Held-out Source: %s", training_sources, held_out_source)

    df_train_pool = df[df[source_col].isin(training_sources)].copy()
    df_held_out = df[df[source_col] == held_out_source].copy()

    # 3. Prepare Feature Matrices
    X_pool_raw, y_pool = separate_features_target(df_train_pool, target_col=BINARY_TARGET_COLUMN)
    X_pool, feature_names, _ = prepare_features(X_pool_raw)

    X_held_raw, y_held = separate_features_target(df_held_out, target_col=BINARY_TARGET_COLUMN)
    X_held, _, _ = prepare_features(X_held_raw)

    # In-domain split (80% train, 20% in-domain test)
    X_train, X_indomain, y_train, y_indomain = split_data(X_pool, y_pool, test_size=0.2, random_seed=42)

    # 4. Train Supervised Baseline (RandomForest & DecisionTree)
    pipelines = build_model_pipelines()
    eval_model_name = "RandomForest" if "RandomForest" in pipelines else list(pipelines.keys())[0]
    pipeline = pipelines[eval_model_name]
    fitted_model, fit_time = train_single_model(pipeline, X_train, y_train, eval_model_name)

    # 5. Evaluate In-Domain vs Held-Out
    metrics_in = evaluate_model(fitted_model, X_indomain, y_indomain, model_name=f"{eval_model_name}_InDomain")
    metrics_out = evaluate_model(fitted_model, X_held, y_held, model_name=f"{eval_model_name}_HeldOut")

    # 6. Evaluate Unsupervised Isolation Forest on Held-Out
    # Train IF strictly on benign training flows
    benign_mask = (y_train == 0)
    X_train_benign = X_train[benign_mask]
    iso_pipe = build_isolation_forest_pipeline(contamination=0.05, random_state=42)
    iso_pipe.fit(X_train_benign)

    scores_held, _, is_anom_held = compute_anomaly_scores(iso_pipe, X_held)
    held_malicious_mask = (y_held == 1)
    held_benign_mask = (y_held == 0)

    iso_mal_detected_pct = float(np.mean(is_anom_held[held_malicious_mask]) * 100.0) if np.sum(held_malicious_mask) > 0 else 0.0
    iso_ben_alarm_pct = float(np.mean(is_anom_held[held_benign_mask]) * 100.0) if np.sum(held_benign_mask) > 0 else 0.0

    # 7. Compute Generalization Disparity
    delta_f1 = round(metrics_in["f1"] - metrics_out["f1"], 4)
    delta_recall = round(metrics_in["recall"] - metrics_out["recall"], 4)
    delta_precision = round(metrics_in["precision"] - metrics_out["precision"], 4)
    delta_accuracy = round(metrics_in["accuracy"] - metrics_out["accuracy"], 4)

    # Per-class breakdown on held-out set if multiclass labels available
    per_class_summary: List[Dict[str, Any]] = []
    if ORIGINAL_LABEL_COLUMN in df_held_out.columns:
        y_held_orig = df_held_out[ORIGINAL_LABEL_COLUMN].astype(str)
        y_train_classes = set(df_train_pool[ORIGINAL_LABEL_COLUMN].astype(str).unique())
        preds_held = fitted_model.predict(X_held)

        for cname in sorted(y_held_orig.unique()):
            c_mask = (y_held_orig == cname)
            c_count = int(np.sum(c_mask))
            c_detected = int(np.sum(preds_held[c_mask] == 1)) if cname.lower() != "benign" else int(np.sum(preds_held[c_mask] == 0))
            rate = round((c_detected / max(1, c_count)) * 100.0, 1)
            is_novel = cname not in y_train_classes
            per_class_summary.append({
                "class_name": cname,
                "count": c_count,
                "is_novel_to_train": is_novel,
                "correctly_classified_rate_pct": rate,
            })

    # 8. Generate Comprehensive Report Markdown
    report_lines = [
        "# AI-NIDS: Held-Out Capture Period & Scenario Generalization Evaluation",
        "",
        "## Executive Summary",
        f"This evaluation rigorously tests model generalization across independent capture periods / scenarios.",
        f"Evaluating machine learning models across temporal periods simulates real-world deployment where traffic",
        f"patterns, attack rates, and protocols shift relative to the training baseline.",
        "",
        "---",
        "",
        "## 1. Experimental Setup",
        f"* **Dataset Source:** `{data_source}`",
        f"* **Training Capture Periods ({len(training_sources)}):** `{', '.join(training_sources)}` ({len(df_train_pool)} flows)",
        f"* **Held-Out Evaluation Period:** `{held_out_source}` ({len(df_held_out)} flows)",
        f"* **Evaluated Supervised Model:** `{eval_model_name}` (class_weight='balanced')",
        f"* **Evaluated Unsupervised Model:** `IsolationForest` (trained exclusively on nominal benign flows)",
        "",
        "---",
        "",
        "## 2. In-Domain vs. Held-Out Generalization Performance",
        "",
        "| Metric | In-Domain Test Set | Held-Out Period (`" + held_out_source + "`) | Generalization Drop (Δ) |",
        "|---|---|---|---|",
        f"| **Accuracy** | {metrics_in['accuracy']:.4f} | {metrics_out['accuracy']:.4f} | {delta_accuracy:+.4f} |",
        f"| **Precision (Malicious)** | {metrics_in['precision']:.4f} | {metrics_out['precision']:.4f} | {delta_precision:+.4f} |",
        f"| **Recall (Malicious)** | {metrics_in['recall']:.4f} | {metrics_out['recall']:.4f} | {delta_recall:+.4f} |",
        f"| **F1-Score (Malicious)** | {metrics_in['f1']:.4f} | {metrics_out['f1']:.4f} | {delta_f1:+.4f} |",
        f"| **ROC-AUC** | {metrics_in.get('roc_auc', 0.0) or 0.0:.4f} | {metrics_out.get('roc_auc', 0.0) or 0.0:.4f} | {((metrics_in.get('roc_auc') or 0.0) - (metrics_out.get('roc_auc') or 0.0)):+.4f} |",
        "",
        "---",
        "",
        "## 3. Unsupervised Outlier Detection on Held-Out Period",
        "* **Isolation Forest Nominal Training Baseline:** Trained on benign traffic only.",
        f"* **Held-Out Malicious Flows Flagged as Outliers:** `{iso_mal_detected_pct:.1f}%`",
        f"* **Held-Out Benign Flows Triggering False Alarms:** `{iso_ben_alarm_pct:.1f}%`",
        "",
        "---",
        "",
        "## 4. Per-Class Behavior on Held-Out Capture",
        "",
        "| Attack Category | Sample Count | Novel to Training? | Detection Rate (%) |",
        "|---|---|---|---|",
    ]

    for item in per_class_summary:
        novelty_tag = "⚠️ **YES (Novel)**" if item["is_novel_to_train"] else "No (Seen)"
        report_lines.append(
            f"| `{item['class_name']}` | {item['count']} | {novelty_tag} | {item['correctly_classified_rate_pct']:.1f}% |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Security & Operational Implications",
        "1. **The Cross-Period Generalization Gap:** Standard random cross-validation often reports over-optimistic",
        "   metrics because flows from the same attack run are split across both train and test partitions.",
        f"   Evaluating on `{held_out_source}` exposes the true domain transfer penalty (Δ F1 = {delta_f1:+.4f}).",
        "2. **Novel Attack Classes:** When a new attack category appears in the held-out period that was absent",
        "   during training, supervised classifiers can misclassify it if its statistical features resemble benign traffic.",
        "3. **Hybrid Defense-in-Depth:** Pairing supervised classifiers with unsupervised Isolation Forest ensures that",
        "   novel, previously unseen attack behaviors still surface high anomaly scores for human analyst investigation.",
        "",
    ])

    report_path = out_dir / "generalization_evaluation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Saved generalization evaluation report → %s", report_path)

    results_dict: Dict[str, Any] = {
        "training_sources": training_sources,
        "held_out_source": held_out_source,
        "model_name": eval_model_name,
        "in_domain_metrics": metrics_in,
        "held_out_metrics": metrics_out,
        "generalization_delta": {
            "delta_f1": delta_f1,
            "delta_recall": delta_recall,
            "delta_precision": delta_precision,
            "delta_accuracy": delta_accuracy,
        },
        "isolation_forest_held_out": {
            "malicious_detected_pct": iso_mal_detected_pct,
            "benign_false_alarm_pct": iso_ben_alarm_pct,
        },
        "per_class_summary": per_class_summary,
        "report_path": str(report_path),
    }

    return results_dict
