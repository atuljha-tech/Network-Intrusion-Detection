"""
multiclass.py — Multiclass Attack Classification Pipeline for AI-NIDS Phase 4.

Implements:
  • Multiclass attack classification using original dataset attack labels.
  • Independent evaluation of each attack class (precision, recall, F1, support).
  • Macro F1, weighted F1, confusion matrices, and ROC-AUC.
  • Class imbalance analysis integration.
  • Feature importance extraction for tree-based classifiers.
  • Winning pipeline persistence and JSON metadata.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    MODELS_DIR,
    MULTICLASS_METADATA_PATH,
    MULTICLASS_PIPELINE_PATH,
    NON_FEATURE_COLUMNS,
    ORIGINAL_LABEL_COLUMN,
    LABEL_COLUMN,
    PROCESSED_CSV,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_SIZE,
    setup_logging,
)
from src.evaluate import (
    compare_multiclass_models,
    evaluate_multiclass_model,
    plot_multiclass_confusion_matrix,
    save_multiclass_reports,
)
from src.explainability import save_feature_importance_reports
from src.feature_engineering import separate_features_target
from src.imbalance import analyze_class_imbalance, save_imbalance_analysis
from src.model_utils import get_package_versions
from src.train import _resolve_data, check_train_test_contamination, prepare_features

logger = setup_logging()
log = logging.getLogger("ai_nids.multiclass")


def build_multiclass_pipelines() -> Dict[str, Pipeline]:
    """Construct baseline multiclass classification pipelines.

    Each pipeline contains median imputation and standard scaling, fitted
    strictly on training data with class_weight='balanced' to counter class imbalance.
    """
    return {
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                class_weight="balanced",
                n_jobs=2,
                max_depth=30,
            )),
        ]),
        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_SEED,
                class_weight="balanced",
                max_depth=25,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
    }


def split_multiclass_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets, using stratification where class counts permit."""
    # Check minimum class size for stratification
    counts = y.value_counts()
    min_count = counts.min()

    if min_count < 2:
        log.warning(
            "Some classes have fewer than 2 instances (%s). Falling back to non-stratified random split.",
            counts[counts < 2].to_dict(),
        )
        return train_test_split(X, y, test_size=test_size, random_state=random_seed)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        log.info("Stratified split: train=%d, test=%d across %d classes", len(X_train), len(X_test), len(counts))
        return X_train, X_test, y_train, y_test
    except ValueError as exc:
        log.warning("Stratified split failed (%s); using random split.", exc)
        return train_test_split(X, y, test_size=test_size, random_state=random_seed)


def run_multiclass_training(
    df: Optional[pd.DataFrame] = None,
    force_synthetic: bool = False,
    test_size: float = TEST_SIZE,
    save_artifacts: bool = True,
) -> Tuple[Dict, str, Dict]:
    """Execute the full multiclass attack classification training pipeline.

    Parameters
    ----------
    df : pd.DataFrame | None
        Input dataset. If None, auto-resolves processed CSV or synthetic data.
    force_synthetic : bool
        If True, forces generation of synthetic data.
    test_size : float
        Fraction held out for testing.
    save_artifacts : bool
        Whether to persist models, plots, and JSON metadata.

    Returns
    -------
    (results, best_model_name, metadata)
    """
    df, data_source = _resolve_data(df, force_synthetic)

    target_col = ORIGINAL_LABEL_COLUMN if ORIGINAL_LABEL_COLUMN in df.columns else LABEL_COLUMN
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found for multiclass classification.")

    log.info("=" * 65)
    log.info("Phase 4 Multiclass Training — source: %s, target: %s", data_source, target_col)
    log.info("Dataset: %d rows × %d columns", len(df), len(df.columns))
    log.info("=" * 65)

    # 1. Feature / target separation
    X, y = separate_features_target(df, target_col=target_col)
    X, feature_names, dropped_cols = prepare_features(X)
    y = y.astype(str)

    unique_classes = sorted(y.unique().tolist())
    log.info("Found %d unique attack classes: %s", len(unique_classes), unique_classes)

    if len(unique_classes) < 2:
        raise ValueError(f"Multiclass classification requires at least 2 classes. Found: {unique_classes}")

    # 2. Split
    X_train, X_test, y_train, y_test = split_multiclass_data(X, y, test_size=test_size)

    # 3. Contamination check
    contamination = check_train_test_contamination(X_train, X_test)

    # 4. Train models
    pipelines = build_multiclass_pipelines()
    results: Dict[str, Dict] = {}

    for name, pipeline in pipelines.items():
        log.info("Training multiclass %s on %d samples...", name, len(X_train))
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        metrics = evaluate_multiclass_model(
            pipeline,
            X_test,
            y_test,
            class_names=unique_classes,
            model_name=name,
        )

        results[name] = {
            "pipeline": pipeline,
            "train_time": train_time,
            "metrics": metrics,
        }

    # 5. Model comparison and selection
    best_name, comparison_df = compare_multiclass_models(results)
    best_pipeline = results[best_name]["pipeline"]
    best_metrics = results[best_name]["metrics"]

    # 6. Class imbalance analysis
    imbalance_analysis = analyze_class_imbalance(
        y_train=y_train,
        y_test=y_test,
        per_class_metrics=best_metrics["per_class_metrics"],
        macro_f1=best_metrics["macro_f1"],
        weighted_f1=best_metrics["weighted_f1"],
        model_name=f"{best_name} (Multiclass)",
    )

    # 7. Persistence and reporting
    if save_artifacts:
        import joblib
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save winning pipeline
        joblib.dump(best_pipeline, MULTICLASS_PIPELINE_PATH)
        log.info("Saved best multiclass pipeline → %s", MULTICLASS_PIPELINE_PATH)

        # Save multiclass reports
        save_multiclass_reports(results)
        save_imbalance_analysis(imbalance_analysis)

        # Plot confusion matrices
        for name, r in results.items():
            plot_multiclass_confusion_matrix(
                r["metrics"]["confusion_matrix"],
                class_names=unique_classes,
                model_name=name,
            )

        # Extract and save feature importances for tree-based models
        for name in ["RandomForest", "DecisionTree"]:
            if name in results:
                try:
                    save_feature_importance_reports(
                        results[name]["pipeline"],
                        feature_names=feature_names,
                        model_name=f"multiclass_{name}",
                    )
                except Exception as exc:
                    log.warning("Could not extract feature importance for %s: %s", name, exc)

    # 8. Build metadata
    train_dist = y_train.value_counts().to_dict()
    test_dist = y_test.value_counts().to_dict()

    metadata: Dict = {
        "model_type": best_name,
        "classification_type": "multiclass",
        "training_date": pd.Timestamp.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "data_source": data_source,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "class_names": unique_classes,
        "n_classes": len(unique_classes),
        "target_column": target_col,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_class_distribution": {str(k): int(v) for k, v in train_dist.items()},
        "test_class_distribution": {str(k): int(v) for k, v in test_dist.items()},
        "contamination": contamination,
        "evaluation_metrics": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in best_metrics.items()
        },
        "all_model_metrics": {
            name: {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in r["metrics"].items()
            }
            for name, r in results.items()
        },
        "imbalance_analysis": imbalance_analysis,
        "package_versions": get_package_versions(),
        "known_limitations": [
            "Severe class imbalance: rare attacks (e.g., Heartbleed, Infiltration) have limited training support.",
            "Higher false-positive confusion between functionally similar attack classes (e.g. DoS Slowloris vs DoS SlowHTTPTest).",
            "Dataset-specific port and protocol distributions in CIC-IDS2017 may not reflect live modern enterprise networks.",
            f"Data source: {data_source}.",
        ],
    }

    if save_artifacts:
        import json
        with open(MULTICLASS_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        log.info("Saved multiclass metadata → %s", MULTICLASS_METADATA_PATH)

    log.info("=" * 65)
    log.info("Multiclass Training Complete. Winning model: %s (Macro F1 = %.4f)", best_name, best_metrics["macro_f1"])
    log.info("=" * 65)

    return results, best_name, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-NIDS Phase 4: Multiclass attack classifier")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()
    run_multiclass_training(force_synthetic=args.synthetic)

