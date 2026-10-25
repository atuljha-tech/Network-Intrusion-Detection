"""
test_phase4.py — Unit and integration tests for AI-NIDS Phase 4.

Tests:
  1. Multiclass attack classification and pipeline persistence.
  2. Independent attack class evaluation (per-class precision, recall, F1, macro/weighted F1).
  3. Class imbalance analysis and metric disparity.
  4. Isolation Forest anomaly detection (separate from supervised classification, score calibration, disclaimer).
  5. Feature importance extraction and visualization for tree-based models.
  6. Explainable AI (SHAP and tree-surrogate attribution outputs).
  7. Common unified prediction interface (binary, multiclass, anomaly, explanations).
  8. Backward compatibility and regression checks.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.anomaly import (
    ANOMALY_DISCLAIMER,
    build_isolation_forest_pipeline,
    compute_anomaly_scores,
    evaluate_anomaly_detection,
    run_anomaly_experiment,
    train_isolation_forest,
)
from src.config import (
    ANOMALY_METADATA_PATH,
    ANOMALY_MODEL_PATH,
    BINARY_TARGET_COLUMN,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    MULTICLASS_METADATA_PATH,
    MULTICLASS_PIPELINE_PATH,
    ORIGINAL_LABEL_COLUMN,
)
from src.evaluate import (
    compare_multiclass_models,
    evaluate_multiclass_model,
    plot_multiclass_confusion_matrix,
    save_multiclass_reports,
)
from src.explainability import (
    FlowExplainer,
    extract_feature_importance,
    plot_feature_importance,
    save_feature_importance_reports,
)
from src.feature_engineering import separate_features_target
from src.imbalance import (
    analyze_class_imbalance,
    calculate_gini_imbalance,
    compute_class_distribution,
    format_imbalance_report,
    save_imbalance_analysis,
)
from src.multiclass import (
    build_multiclass_pipelines,
    run_multiclass_training,
    split_multiclass_data,
)
from src.predict import UnifiedPredictor, predict, predict_single
from src.run_eda import _generate_synthetic_data
from src.train import prepare_features


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_multiclass_data():
    """Generate a clean synthetic multiclass dataset for testing."""
    df = _generate_synthetic_data(n=300)
    X, y = separate_features_target(df, target_col=ORIGINAL_LABEL_COLUMN)
    X, feature_names, _ = prepare_features(X)
    return X, y.astype(str), feature_names, df


@pytest.fixture
def fitted_multiclass_model(synthetic_multiclass_data):
    """Provide a fitted multiclass DecisionTree pipeline."""
    X, y, _, _ = synthetic_multiclass_data
    pipelines = build_multiclass_pipelines()
    dt = pipelines["DecisionTree"]
    dt.fit(X, y)
    return dt


# ── Test Multiclass Pipeline & Evaluation (Tasks 1, 2, 3) ───────────────────

def test_multiclass_pipeline_construction():
    pipelines = build_multiclass_pipelines()
    assert "RandomForest" in pipelines
    assert "DecisionTree" in pipelines
    assert "LogisticRegression" in pipelines

    for name, pipe in pipelines.items():
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")


def test_independent_attack_class_evaluation(synthetic_multiclass_data, fitted_multiclass_model):
    X, y, feature_names, _ = synthetic_multiclass_data
    classes = sorted(y.unique().tolist())

    metrics = evaluate_multiclass_model(
        fitted_multiclass_model,
        X,
        y,
        class_names=classes,
        model_name="DecisionTreeTest",
    )

    # Global metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0

    # Per-class metrics
    per_class = metrics["per_class_metrics"]
    for c in classes:
        assert c in per_class
        assert "precision" in per_class[c]
        assert "recall" in per_class[c]
        assert "f1" in per_class[c]
        assert "support" in per_class[c]
        assert 0.0 <= per_class[c]["precision"] <= 1.0
        assert 0.0 <= per_class[c]["recall"] <= 1.0
        assert 0.0 <= per_class[c]["f1"] <= 1.0
        assert per_class[c]["support"] >= 0

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    assert cm.shape == (len(classes), len(classes))
    assert cm.sum() == len(X)


def test_compare_multiclass_models():
    mock_results = {
        "ModelA": {
            "pipeline": None,
            "train_time": 1.0,
            "metrics": {
                "accuracy": 0.85,
                "macro_f1": 0.50,
                "weighted_f1": 0.80,
                "macro_precision": 0.55,
                "weighted_precision": 0.82,
                "macro_recall": 0.52,
                "weighted_recall": 0.85,
                "predict_time": 0.05,
            },
        },
        "ModelB": {
            "pipeline": None,
            "train_time": 1.2,
            "metrics": {
                "accuracy": 0.75,
                "macro_f1": 0.70,  # Higher macro F1
                "weighted_f1": 0.78,
                "macro_precision": 0.72,
                "weighted_precision": 0.80,
                "macro_recall": 0.69,
                "weighted_recall": 0.75,
                "predict_time": 0.06,
            },
        },
    }

    best_name, comp_df = compare_multiclass_models(mock_results)
    # ModelB should be selected based on highest Macro F1
    assert best_name == "ModelB"
    assert len(comp_df) == 2


def test_plot_multiclass_confusion_matrix(tmp_path):
    cm = np.array([[50, 2], [5, 43]])
    class_names = ["benign", "ddos"]
    out_path = plot_multiclass_confusion_matrix(
        cm, class_names, "TestModel", output_dir=tmp_path
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# ── Test Class Imbalance Analysis (Task 4) ──────────────────────────────────

def test_class_imbalance_distribution():
    y = pd.Series(["benign"] * 80 + ["ddos"] * 15 + ["heartbleed"] * 5)
    dist_df = compute_class_distribution(y)

    assert len(dist_df) == 3
    assert dist_df.iloc[0]["class_name"] == "benign"
    assert dist_df.iloc[0]["count"] == 80
    assert dist_df.iloc[0]["percentage"] == 80.0
    assert dist_df.iloc[-1]["class_name"] == "heartbleed"
    assert dist_df.iloc[-1]["count"] == 5
    assert dist_df.iloc[-1]["imbalance_ratio_vs_max"] == 16.0


def test_gini_imbalance_calculation():
    # Equal distribution -> Gini near 0
    equal_counts = np.array([100, 100, 100, 100])
    gini_equal = calculate_gini_imbalance(equal_counts)
    assert gini_equal == pytest.approx(0.0, abs=0.01)

    # Highly imbalanced distribution -> Gini high
    skewed_counts = np.array([10000, 10, 5, 1])
    gini_skewed = calculate_gini_imbalance(skewed_counts)
    assert gini_skewed > 0.70


def test_analyze_class_imbalance_report(tmp_path):
    y_train = pd.Series(["benign"] * 900 + ["ddos"] * 90 + ["infiltration"] * 10)
    per_class = {
        "benign": {"precision": 0.95, "recall": 0.99, "f1": 0.97, "support": 900},
        "ddos": {"precision": 0.85, "recall": 0.75, "f1": 0.80, "support": 90},
        "infiltration": {"precision": 0.20, "recall": 0.10, "f1": 0.13, "support": 10},
    }

    analysis = analyze_class_imbalance(
        y_train,
        per_class_metrics=per_class,
        macro_f1=0.633,
        weighted_f1=0.946,
        model_name="ImbalanceTest",
    )

    assert analysis["majority_class"] == "benign"
    assert analysis["minority_class"] == "infiltration"
    assert analysis["imbalance_ratio"] == 90.0
    assert "infiltration" in analysis["extreme_minority_classes_lt_1pct"]
    assert analysis["metric_disparity_weighted_minus_macro"] == pytest.approx(0.313, abs=0.01)

    # Report text formatting
    report_text = format_imbalance_report(analysis)
    assert "Imbalance Ratio:" in report_text
    assert "infiltration" in report_text

    # File saving
    json_p, txt_p = save_imbalance_analysis(analysis, output_dir=tmp_path)
    assert json_p.exists()
    assert txt_p.exists()


# ── Test Isolation Forest Anomaly Detection (Tasks 5, 6) ───────────────────

def test_isolation_forest_training_and_scoring(synthetic_multiclass_data):
    X, _, _, df = synthetic_multiclass_data
    y_bin = df[BINARY_TARGET_COLUMN]

    pipeline, info = train_isolation_forest(
        X, y_train_binary=y_bin, fit_on_benign_only=True, n_estimators=50
    )

    assert "disclaimer" in info
    assert info["training_mode"] == "semi_supervised_benign_baseline"

    calibrated_scores, raw_scores, is_outlier = compute_anomaly_scores(pipeline, X)

    assert len(calibrated_scores) == len(X)
    assert len(raw_scores) == len(X)
    assert len(is_outlier) == len(X)

    # Calibrated scores must be bounded strictly in [0.0, 1.0]
    assert np.all(calibrated_scores >= 0.0)
    assert np.all(calibrated_scores <= 1.0)
    assert is_outlier.dtype == bool


def test_anomaly_detection_strict_separation_from_attack_labels(synthetic_multiclass_data):
    """CRITICAL REQUIREMENT (Task 6): Do NOT treat anomaly scores as confirmed attack labels."""
    X, y_multi, _, df = synthetic_multiclass_data
    y_bin = df[BINARY_TARGET_COLUMN]

    pipeline, _ = train_isolation_forest(X, y_train_binary=y_bin, n_estimators=50)
    results = evaluate_anomaly_detection(pipeline, X, y_bin, y_test_multiclass=y_multi)

    # Must contain explicit security disclaimer
    assert "disclaimer" in results
    assert results["disclaimer"] == ANOMALY_DISCLAIMER
    assert "DO NOT represent confirmed malicious attacks" in results["disclaimer"]

    # False alarm rate on benign traffic must be explicitly measured
    assert "benign_false_alarm_rate_pct" in results
    assert "malicious_outlier_rate_pct" in results
    assert "per_class_outlier_analysis" in results


def test_run_anomaly_experiment_end_to_end(tmp_path, synthetic_multiclass_data):
    X, y_multi, feature_names, df = synthetic_multiclass_data
    y_bin = df[BINARY_TARGET_COLUMN]

    X_train, X_test = X.iloc[:200], X.iloc[200:]
    y_train_bin, y_test_bin = y_bin.iloc[:200], y_bin.iloc[200:]
    y_test_multi = y_multi.iloc[200:]

    with patch("src.anomaly.MODELS_DIR", tmp_path), patch("src.anomaly.REPORTS_DIR", tmp_path), patch("src.anomaly.ANOMALY_MODEL_PATH", tmp_path / "iso.joblib"), patch("src.anomaly.ANOMALY_METADATA_PATH", tmp_path / "meta.json"):
        pipeline, info, results = run_anomaly_experiment(
            X_train,
            y_train_bin,
            X_test,
            y_test_bin,
            y_test_multiclass=y_test_multi,
            feature_names=feature_names,
            save_artifacts=True,
        )

        assert (tmp_path / "iso.joblib").exists()
        assert (tmp_path / "meta.json").exists()
        assert (tmp_path / "anomaly_detection_report.txt").exists()


# ── Test Feature Importance (Task 7) ────────────────────────────────────────

def test_extract_feature_importance(synthetic_multiclass_data, fitted_multiclass_model):
    X, _, feature_names, _ = synthetic_multiclass_data

    imp_df = extract_feature_importance(fitted_multiclass_model, feature_names)

    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns
    assert "relative_importance_pct" in imp_df.columns
    assert "cumulative_importance_pct" in imp_df.columns

    # Verify descending sort order
    assert imp_df["importance"].is_monotonic_decreasing

    # Relative importance should sum to approx 100%
    assert imp_df["relative_importance_pct"].sum() == pytest.approx(100.0, abs=0.1)
    assert imp_df["cumulative_importance_pct"].iloc[-1] == pytest.approx(100.0, abs=0.1)


def test_extract_feature_importance_invalid_model():
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    with pytest.raises(AttributeError, match="feature_importances_"):
        extract_feature_importance(lr, ["feat_a", "feat_b"])


def test_plot_and_save_feature_importance(tmp_path, synthetic_multiclass_data, fitted_multiclass_model):
    _, _, feature_names, _ = synthetic_multiclass_data

    csv_p, png_p = save_feature_importance_reports(
        fitted_multiclass_model,
        feature_names=feature_names,
        model_name="UnitTestModel",
        output_dir=tmp_path,
    )
    assert csv_p.exists()
    assert png_p.exists()


# ── Test Explainability & SHAP / Surrogate Fallback (Task 8) ────────────────

def test_flow_explainer_surrogate_fallback(synthetic_multiclass_data, fitted_multiclass_model):
    X, _, feature_names, _ = synthetic_multiclass_data

    explainer = FlowExplainer(
        fitted_multiclass_model,
        feature_names=feature_names,
        background_data=X.iloc[:50],
    )

    sample = X.iloc[0].to_dict()
    explanation = explainer.explain_instance(sample, top_n=5, predicted_class="ddos")

    assert "method" in explanation
    assert "top_features" in explanation
    assert len(explanation["top_features"]) == 5
    assert "summary" in explanation

    first_feat = explanation["top_features"][0]
    assert "feature" in first_feat
    assert "attribution" in first_feat
    assert "feature_value" in first_feat
    assert "impact_direction" in first_feat


def test_flow_explainer_with_mocked_shap(synthetic_multiclass_data, fitted_multiclass_model):
    X, _, feature_names, _ = synthetic_multiclass_data

    # Mock shap module to test SHAP code path
    mock_shap_values = MagicMock()
    mock_shap_values.values = np.ones((1, len(feature_names))) * 0.25

    with patch("src.explainability.SHAP_AVAILABLE", True), patch("src.explainability.shap") as mock_shap:
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer_instance

        explainer = FlowExplainer(
            fitted_multiclass_model,
            feature_names=feature_names,
            background_data=X.iloc[:20],
        )

        explanation = explainer.explain_instance(X.iloc[0], top_n=3, predicted_class="portscan")
        assert explanation["method"] == "shap_tree_explainer"
        assert explanation["shap_available"] is True
        assert len(explanation["top_features"]) == 3


# ── Test Common Prediction Interface (Tasks 9, 10) ──────────────────────────

def test_unified_predictor_binary_and_multiclass(synthetic_multiclass_data):
    X, y_multi, feature_names, df = synthetic_multiclass_data
    y_bin = df[BINARY_TARGET_COLUMN]

    # Train a minimal binary model
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ])
    binary_pipe.fit(X, y_bin)

    multi_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ])
    multi_pipe.fit(X, y_multi)

    binary_meta = {
        "feature_names": feature_names,
        "classification_type": "binary",
    }
    multi_meta = {
        "feature_names": feature_names,
        "classification_type": "multiclass",
        "class_names": sorted(y_multi.unique().tolist()),
    }

    # Test predict() with binary model
    res_bin = predict(X.head(5), pipeline=binary_pipe, metadata=binary_meta)
    assert "prediction" in res_bin.columns
    assert "label" in res_bin.columns
    assert "probability_malicious" in res_bin.columns
    assert "probability_benign" in res_bin.columns

    # Test predict() with multiclass model
    res_multi = predict(X.head(5), pipeline=multi_pipe, metadata=multi_meta)
    assert "prediction" in res_multi.columns
    assert "predicted_class" in res_multi.columns
    assert "score" in res_multi.columns
    assert "probabilities" in res_multi.columns

    # Test predict_single()
    single_res = predict_single(X.iloc[0].to_dict(), pipeline=multi_pipe, metadata=multi_meta)
    assert "prediction" in single_res
    assert "predicted_class" in single_res
    assert "score" in single_res


def test_predict_with_anomaly_and_explanation(synthetic_multiclass_data, fitted_multiclass_model):
    X, _, feature_names, _ = synthetic_multiclass_data

    iso_pipe = build_isolation_forest_pipeline(n_estimators=20)
    iso_pipe.fit(X)

    metadata = {
        "feature_names": feature_names,
        "classification_type": "multiclass",
        "class_names": ["benign", "bot", "ddos"],
    }

    res = predict(
        X.head(3),
        pipeline=fitted_multiclass_model,
        metadata=metadata,
        include_anomaly=True,
        anomaly_pipeline=iso_pipe,
        include_explanation=True,
        top_n_features=3,
    )

    assert "anomaly_score" in res.columns
    assert "is_anomaly" in res.columns
    assert "explanation" in res.columns
    assert len(res["explanation"].iloc[0]["top_features"]) == 3


def test_unified_predictor_class(synthetic_multiclass_data):
    X, _, feature_names, _ = synthetic_multiclass_data

    predictor = UnifiedPredictor(auto_load=False)
    predictor.binary_pipeline = build_multiclass_pipelines()["DecisionTree"]
    predictor.binary_pipeline.fit(X, (X["flow_duration"] > 50000).astype(int))
    predictor.binary_metadata = {"feature_names": feature_names, "classification_type": "binary"}

    # Single dict inference
    single_out = predictor.predict(X.iloc[0].to_dict(), model_type="binary")
    assert isinstance(single_out, dict)
    assert single_out["model_type"] == "binary"
    assert "label" in single_out

    # Batch DataFrame inference
    batch_out = predictor.predict(X.head(5), model_type="binary")
    assert isinstance(batch_out, pd.DataFrame)
    assert len(batch_out) == 5

