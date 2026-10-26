"""
predict.py — Common Prediction and Inference Interface for AI-NIDS.

Supports:
  • Binary intrusion detection (Benign vs Malicious).
  • Multiclass attack classification using original CIC-IDS2017 categories.
  • Unsupervised Isolation Forest anomaly scoring and statistical outlier triage.
  • Explainable AI (XAI) feature attributions (SHAP / tree surrogate).
  • Backward-compatible batch DataFrame and single-flow prediction APIs.

These predictions operate on dataset records or explicitly supplied feature rows.
This module does NOT perform live network intrusion detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.anomaly import compute_anomaly_scores
from src.config import (
    ANOMALY_METADATA_PATH,
    ANOMALY_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    MULTICLASS_METADATA_PATH,
    MULTICLASS_PIPELINE_PATH,
    setup_logging,
)
from src.explainability import FlowExplainer
from src.model_utils import (
    load_metadata,
    load_pipeline,
    patch_sklearn_compatibility,
    validate_features,
)

logger = setup_logging()
log = logging.getLogger("ai_nids.predict")


# ── Core Prediction Engine ───────────────────────────────────────────────────

def predict(
    X: pd.DataFrame,
    pipeline: Any = None,
    metadata: Optional[Dict] = None,
    pipeline_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    include_anomaly: bool = False,
    anomaly_pipeline: Any = None,
    anomaly_metadata: Optional[Dict] = None,
    include_explanation: bool = False,
    top_n_features: int = 5,
) -> pd.DataFrame:
    """Run predictions on input features using a saved or provided pipeline.

    Maintains 100% backward-compatibility with Phase 3 binary predictions
    while seamlessly supporting multiclass classification, anomaly scoring,
    and feature explanations.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.
    pipeline : sklearn Pipeline | None
        Fitted classification pipeline. Loaded from disk if None.
    metadata : dict | None
        Model metadata. Loaded from disk if None.
    pipeline_path / metadata_path : Path | None
        Custom file paths for classification pipeline / metadata.
    include_anomaly : bool
        If True, appends 'anomaly_score' and 'is_anomaly' columns.
    anomaly_pipeline / anomaly_metadata : Any | None
        Optional pre-loaded anomaly model and metadata.
    include_explanation : bool
        If True, appends an 'explanation' column with top contributing factors.
    top_n_features : int
        Number of top contributing features in explanation.

    Returns
    -------
    pd.DataFrame containing predictions, scores/probabilities, and optional XAI outputs.
    """
    if pipeline is None:
        pipeline = load_pipeline(pipeline_path or MODEL_PIPELINE_PATH)
    else:
        pipeline = patch_sklearn_compatibility(pipeline)

    if anomaly_pipeline is not None:
        anomaly_pipeline = patch_sklearn_compatibility(anomaly_pipeline)
    if metadata is None:
        metadata = load_metadata(metadata_path or MODEL_METADATA_PATH)

    feature_names = metadata["feature_names"]
    is_multiclass = metadata.get("classification_type") == "multiclass"

    # Replace ±inf with NaN
    X_clean = X.replace([np.inf, -np.inf], np.nan)

    # Validate schema
    X_validated = validate_features(X_clean, feature_names)

    # Handle empty DataFrame gracefully
    if len(X_validated) == 0:
        if not is_multiclass:
            empty_cols = ["prediction", "label", "probability_malicious", "probability_benign", "score"]
        else:
            empty_cols = ["prediction", "predicted_class", "label", "score", "probabilities"]
        if include_anomaly:
            empty_cols.extend(["anomaly_score", "is_anomaly"])
        if include_explanation:
            empty_cols.append("explanation")
        res_empty = pd.DataFrame(columns=empty_cols, index=X.index)
        log.info("Predicted 0 sample(s) on empty input.")
        return res_empty

    # Classification predictions
    raw_preds = pipeline.predict(X_validated)

    if not is_multiclass:
        # Binary Classification (Phase 3 schema)
        result = pd.DataFrame(
            {
                "prediction": raw_preds,
                "label": np.where(raw_preds == 0, "Benign", "Malicious"),
            },
            index=X.index,
        )

        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_validated)
                result["probability_malicious"] = proba[:, 1]
                result["probability_benign"] = proba[:, 0]
                result["score"] = np.max(proba, axis=1)
            except Exception as exc:
                log.warning("Could not compute binary probabilities: %s", exc)
        elif hasattr(pipeline, "decision_function"):
            try:
                result["score"] = pipeline.decision_function(X_validated)
            except Exception:
                pass
    else:
        # Multiclass Classification (Phase 4 schema)
        class_names = metadata.get("class_names")
        if class_names is None and hasattr(pipeline, "classes_"):
            class_names = [str(c) for c in pipeline.classes_]

        preds_str = [str(p) for p in raw_preds]
        result = pd.DataFrame(
            {
                "prediction": raw_preds,
                "predicted_class": preds_str,
                "label": preds_str,
            },
            index=X.index,
        )

        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_validated)
                result["score"] = np.max(proba, axis=1)

                # Store per-class probability mapping
                if class_names and len(class_names) == proba.shape[1]:
                    prob_records = []
                    for row in proba:
                        prob_records.append({str(c): round(float(p), 4) for c, p in zip(class_names, row)})
                    result["probabilities"] = prob_records

                    # For convenience, add individual columns if few classes
                    if len(class_names) <= 10:
                        for idx, c in enumerate(class_names):
                            result[f"probability_{c}"] = proba[:, idx]
            except Exception as exc:
                log.warning("Could not compute multiclass probabilities: %s", exc)

    # Optional Anomaly Detection
    if include_anomaly:
        try:
            if anomaly_pipeline is None:
                anomaly_pipeline = load_pipeline(ANOMALY_MODEL_PATH)
            calibrated_scores, _, is_outlier = compute_anomaly_scores(anomaly_pipeline, X_validated)
            result["anomaly_score"] = calibrated_scores
            result["is_anomaly"] = is_outlier
        except Exception as exc:
            log.warning("Could not compute anomaly scores: %s", exc)

    # Optional Explanation
    if include_explanation:
        try:
            explainer = FlowExplainer(pipeline, feature_names=feature_names)
            explanations = []
            for i in range(len(X_validated)):
                row_pred = result["label"].iloc[i] if "label" in result.columns else None
                exp_dict = explainer.explain_instance(
                    X_validated.iloc[i],
                    top_n=top_n_features,
                    predicted_class=row_pred,
                )
                explanations.append(exp_dict)
            result["explanation"] = explanations
        except Exception as exc:
            log.warning("Could not compute explanations: %s", exc)

    log.info("Predicted %d sample(s).", len(result))
    return result


def predict_single(
    features: Dict[str, float],
    pipeline: Any = None,
    metadata: Optional[Dict] = None,
    include_anomaly: bool = False,
    include_explanation: bool = False,
) -> Dict[str, Any]:
    """Predict a single sample from a dictionary of feature values.

    Parameters
    ----------
    features : dict
        Mapping of feature_name -> float value.
    pipeline / metadata : Any | None
        Optional model and metadata.

    Returns
    -------
    dict with prediction, label, score/probabilities, and optional anomaly / explanation.
    """
    X = pd.DataFrame([features])
    result = predict(
        X,
        pipeline=pipeline,
        metadata=metadata,
        include_anomaly=include_anomaly,
        include_explanation=include_explanation,
    )
    return result.iloc[0].to_dict()


# ── Unified Predictor Interface ──────────────────────────────────────────────

class UnifiedPredictor:
    """Common prediction interface managing binary, multiclass, and anomaly models.

    Provides a clean, unified API for inference across model paradigms,
    returning standardized prediction payloads with model confidence,
    calibrated anomaly metrics, and local explanations.
    """

    def __init__(
        self,
        binary_pipeline_path: Optional[Path] = None,
        binary_metadata_path: Optional[Path] = None,
        multiclass_pipeline_path: Optional[Path] = None,
        multiclass_metadata_path: Optional[Path] = None,
        anomaly_pipeline_path: Optional[Path] = None,
        anomaly_metadata_path: Optional[Path] = None,
        auto_load: bool = True,
    ):
        self.binary_pipeline_path = binary_pipeline_path or MODEL_PIPELINE_PATH
        self.binary_metadata_path = binary_metadata_path or MODEL_METADATA_PATH
        self.multiclass_pipeline_path = multiclass_pipeline_path or MULTICLASS_PIPELINE_PATH
        self.multiclass_metadata_path = multiclass_metadata_path or MULTICLASS_METADATA_PATH
        self.anomaly_pipeline_path = anomaly_pipeline_path or ANOMALY_MODEL_PATH
        self.anomaly_metadata_path = anomaly_metadata_path or ANOMALY_METADATA_PATH

        self.binary_pipeline = None
        self.binary_metadata = None
        self.multiclass_pipeline = None
        self.multiclass_metadata = None
        self.anomaly_pipeline = None
        self.anomaly_metadata = None

        if auto_load:
            self.load_models()

    def load_models(self) -> None:
        """Attempt to load available models and metadata from disk."""
        if self.binary_pipeline_path.exists() and self.binary_metadata_path.exists():
            try:
                self.binary_pipeline = load_pipeline(self.binary_pipeline_path)
                self.binary_metadata = load_metadata(self.binary_metadata_path)
                log.info("UnifiedPredictor: Binary model loaded.")
            except Exception as exc:
                log.warning("Could not load binary model: %s", exc)

        if self.multiclass_pipeline_path.exists() and self.multiclass_metadata_path.exists():
            try:
                self.multiclass_pipeline = load_pipeline(self.multiclass_pipeline_path)
                self.multiclass_metadata = load_metadata(self.multiclass_metadata_path)
                log.info("UnifiedPredictor: Multiclass model loaded.")
            except Exception as exc:
                log.warning("Could not load multiclass model: %s", exc)

        if self.anomaly_pipeline_path.exists() and self.anomaly_metadata_path.exists():
            try:
                self.anomaly_pipeline = load_pipeline(self.anomaly_pipeline_path)
                self.anomaly_metadata = load_metadata(self.anomaly_metadata_path)
                log.info("UnifiedPredictor: Anomaly model loaded.")
            except Exception as exc:
                log.warning("Could not load anomaly model: %s", exc)

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, float], List[Dict[str, float]]],
        model_type: str = "binary",
        include_anomaly: bool = False,
        include_explanation: bool = False,
        top_n_features: int = 5,
    ) -> Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
        """Execute prediction through the common unified interface.

        Parameters
        ----------
        data : DataFrame, dict, or list of dicts
            Input feature observation(s).
        model_type : "binary" | "multiclass"
            Which model paradigm to query.
        include_anomaly : bool
            Whether to compute unsupervised Isolation Forest anomaly scores.
        include_explanation : bool
            Whether to compute feature attribution explanations.

        Returns
        -------
        DataFrame (if DataFrame input) or Dict/List[Dict] (if single dict or list of dicts).
        """
        is_single = isinstance(data, dict)
        if is_single:
            df_in = pd.DataFrame([data])
        elif isinstance(data, list):
            df_in = pd.DataFrame(data)
        else:
            df_in = data.copy()

        if model_type == "multiclass":
            if self.multiclass_pipeline is None:
                raise RuntimeError(
                    f"Multiclass pipeline not loaded. Train a multiclass model or ensure {self.multiclass_pipeline_path} exists."
                )
            pipeline = self.multiclass_pipeline
            metadata = self.multiclass_metadata
        elif model_type == "binary":
            if self.binary_pipeline is None:
                raise RuntimeError(
                    f"Binary pipeline not loaded. Train a binary model or ensure {self.binary_pipeline_path} exists."
                )
            pipeline = self.binary_pipeline
            metadata = self.binary_metadata
        else:
            raise ValueError(f"Invalid model_type '{model_type}'. Allowed choices: 'binary', 'multiclass'")

        res_df = predict(
            df_in,
            pipeline=pipeline,
            metadata=metadata,
            include_anomaly=include_anomaly,
            anomaly_pipeline=self.anomaly_pipeline,
            anomaly_metadata=self.anomaly_metadata,
            include_explanation=include_explanation,
            top_n_features=top_n_features,
        )

        res_df["model_type"] = model_type

        if is_single:
            return res_df.iloc[0].to_dict()
        return res_df
