"""
explainability.py — Explainable AI (XAI) for AI-NIDS Phase 4.

Provides:
  1. Feature importance extraction and visualization for tree-based models
     (Random Forest, Decision Tree).
  2. SHAP (SHapley Additive exPlanations) integration where compatible with
     the local environment and resource constraints.
  3. Robust, deterministic tree-attribution surrogate fallback when SHAP is
     unavailable or computationally constrained.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import REPORTS_DIR, setup_logging

logger = setup_logging()
log = logging.getLogger("ai_nids.explainability")

# Check SHAP availability in current environment
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
    log.info("SHAP library is available in environment.")
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    log.info("SHAP library is not installed in current environment. Using tree attribution fallback.")


# ── Feature Importance (Tree-Based Models) ──────────────────────────────────

def extract_classifier_from_pipeline(pipeline: Any) -> Any:
    """Extract underlying classifier estimator from an sklearn Pipeline."""
    if hasattr(pipeline, "named_steps"):
        if "classifier" in pipeline.named_steps:
            return pipeline.named_steps["classifier"]
        # Fall back to final step
        return pipeline.steps[-1][1]
    return pipeline


def extract_feature_importance(
    pipeline: Any,
    feature_names: List[str],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Extract feature importance from a tree-based classifier.

    Parameters
    ----------
    pipeline : Pipeline or Classifier
        Fitted model containing ``feature_importances_``.
    feature_names : list of str
        Names of the features in order of model input.
    top_n : int | None
        If provided, limits return to top N features.

    Returns
    -------
    pd.DataFrame with columns:
        * ``feature``
        * ``importance``
        * ``relative_importance_pct``
        * ``cumulative_importance_pct``
    """
    clf = extract_classifier_from_pipeline(pipeline)

    if not hasattr(clf, "feature_importances_"):
        raise AttributeError(
            f"Model {type(clf).__name__} does not have 'feature_importances_' attribute. "
            f"Feature importance requires tree-based models (e.g., RandomForest, DecisionTree)."
        )

    importances = clf.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Number of feature importances ({len(importances)}) does not match "
            f"number of feature names ({len(feature_names)})."
        )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    # Sort descending
    df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    total_imp = df["importance"].sum()
    if total_imp > 0:
        df["relative_importance_pct"] = (df["importance"] / total_imp) * 100.0
    else:
        df["relative_importance_pct"] = 0.0

    df["cumulative_importance_pct"] = df["relative_importance_pct"].cumsum()

    if top_n is not None and top_n > 0:
        df = df.head(top_n).copy()

    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate and save a horizontal bar chart of top feature importances."""
    output_path = output_path or (REPORTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not MATPLOTLIB_AVAILABLE or plt is None:
        log.warning("matplotlib not available; skipping plot generation.")
        return output_path

    plot_df = importance_df.head(top_n).sort_values(by="importance", ascending=True)

    figsize = (8, max(5.0, len(plot_df) * 0.35))
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(plot_df["feature"], plot_df["relative_importance_pct"], color="#1f77b4", edgecolor="none")
    ax.set_xlabel("Relative Importance (%)", fontsize=10)
    ax.set_title(f"Top {len(plot_df)} Feature Importances — {model_name}", fontsize=11, fontweight="bold")

    # Annotate bar percentages
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=8,
            color="#333333",
        )

    ax.set_xlim(0, max(plot_df["relative_importance_pct"].max() * 1.15, 5.0))
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved feature importance plot → %s", output_path)
    return output_path


def save_feature_importance_reports(
    pipeline: Any,
    feature_names: List[str],
    model_name: str,
    output_dir: Optional[Path] = None,
    top_n: int = 20,
) -> Tuple[Path, Path]:
    """Extract, save CSV, and plot feature importance for a tree-based model."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    csv_path = output_dir / f"feature_importance_{safe_name}.csv"
    png_path = output_dir / f"feature_importance_{safe_name}.png"

    df_full = extract_feature_importance(pipeline, feature_names)
    df_full.to_csv(csv_path, index=False)
    log.info("Saved feature importance CSV → %s", csv_path)

    plot_feature_importance(df_full, model_name=model_name, top_n=top_n, output_path=png_path)
    return csv_path, png_path


# ── SHAP Integration & Attribution Fallback ──────────────────────────────────

class FlowExplainer:
    """Unified explainer supporting SHAP TreeExplainer and robust surrogate fallback.

    Parameters
    ----------
    pipeline : Pipeline or Classifier
        Fitted model to explain.
    feature_names : list of str
        Ordered list of feature column names.
    background_data : pd.DataFrame | None
        Representative background reference dataset for SHAP or surrogate medians.
        If large, will be sampled to bound computation to <= 100 rows.
    """

    def __init__(
        self,
        pipeline: Any,
        feature_names: List[str],
        background_data: Optional[pd.DataFrame] = None,
        max_background_samples: int = 100,
    ):
        self.pipeline = pipeline
        self.classifier = extract_classifier_from_pipeline(pipeline)
        self.feature_names = list(feature_names)
        self.max_background_samples = max_background_samples

        # Prepare reference background statistics
        if background_data is not None and not background_data.empty:
            bg_clean = background_data[self.feature_names].replace([np.inf, -np.inf], np.nan)
            if len(bg_clean) > self.max_background_samples:
                self.background_sample = bg_clean.sample(
                    n=self.max_background_samples, random_state=42
                )
            else:
                self.background_sample = bg_clean.copy()

            self.reference_medians = bg_clean.median().to_dict()
            self.reference_iqrs = (bg_clean.quantile(0.75) - bg_clean.quantile(0.25)).to_dict()
        else:
            self.background_sample = None
            self.reference_medians = {f: 0.0 for f in self.feature_names}
            self.reference_iqrs = {f: 1.0 for f in self.feature_names}

        # Cache tree feature importances if tree-based
        if hasattr(self.classifier, "feature_importances_"):
            self.tree_importances = dict(
                zip(self.feature_names, self.classifier.feature_importances_)
            )
        else:
            self.tree_importances = {f: 1.0 / len(self.feature_names) for f in self.feature_names}

        # Initialize SHAP if available
        self.shap_explainer = None
        self.shap_init_error = None
        if SHAP_AVAILABLE:
            try:
                # TreeExplainer for tree models
                if hasattr(self.classifier, "feature_importances_"):
                    self.shap_explainer = shap.TreeExplainer(
                        self.classifier,
                        data=self.background_sample,
                        feature_perturbation="interventional" if self.background_sample is not None else "tree_path_dependent",
                    )
                else:
                    self.shap_explainer = shap.Explainer(
                        self.classifier,
                        self.background_sample,
                    )
                log.info("Initialized SHAP TreeExplainer successfully.")
            except Exception as exc:
                self.shap_init_error = str(exc)
                log.warning("SHAP initialization failed: %s. Using surrogate attribution.", exc)

    def explain_instance(
        self,
        X_row: Union[pd.Series, pd.DataFrame, Dict[str, float]],
        top_n: int = 5,
        predicted_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a feature attribution explanation for a single flow sample.

        Returns a standardized explanation dictionary regardless of whether SHAP
        or the surrogate attribution engine produced it.
        """
        # Convert row to Series
        if isinstance(X_row, dict):
            row_series = pd.Series(X_row)
        elif isinstance(X_row, pd.DataFrame):
            row_series = X_row.iloc[0]
        else:
            row_series = X_row

        # Try SHAP first if available
        if self.shap_explainer is not None:
            try:
                row_vals = row_series.reindex(self.feature_names).fillna(0.0).values.reshape(1, -1)
                shap_res = self.shap_explainer(row_vals)
                shap_vals = shap_res.values[0]

                # If multiclass output (dimensions: [classes, features] or [features, classes])
                if len(shap_vals.shape) > 1:
                    # Take slice for predicted class or highest magnitude
                    shap_vals = shap_vals[:, 0] if shap_vals.shape[1] > 1 else shap_vals.ravel()

                attributions = []
                for idx, fname in enumerate(self.feature_names):
                    val = float(row_series.get(fname, 0.0))
                    attr = float(shap_vals[idx])
                    attributions.append({
                        "feature": fname,
                        "attribution": round(attr, 4),
                        "abs_attribution": abs(attr),
                        "feature_value": val,
                        "impact_direction": "increases_anomaly_risk" if attr > 0 else "decreases_anomaly_risk",
                    })

                attributions.sort(key=lambda x: x["abs_attribution"], reverse=True)
                top_features = attributions[:top_n]

                return {
                    "method": "shap_tree_explainer",
                    "shap_available": True,
                    "status": "success",
                    "predicted_class": str(predicted_class) if predicted_class is not None else "Unknown",
                    "top_features": top_features,
                    "summary": _build_explanation_summary(top_features, predicted_class),
                }
            except Exception as exc:
                log.warning("SHAP instance explanation failed: %s. Falling back to surrogate.", exc)

        # Fallback: Tree-weighted deviation surrogate
        return self._explain_surrogate(row_series, top_n=top_n, predicted_class=predicted_class)

    def _explain_surrogate(
        self,
        row_series: pd.Series,
        top_n: int = 5,
        predicted_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute local attribution using tree feature importances combined with baseline deviation."""
        attributions = []
        for fname in self.feature_names:
            val = float(row_series.get(fname, 0.0))
            if np.isnan(val) or np.isinf(val):
                val = 0.0

            med = float(self.reference_medians.get(fname, 0.0))
            iqr = float(self.reference_iqrs.get(fname, 1.0))
            scale = max(iqr, 1e-4)

            # Standardized robust deviation (Z-score via median and IQR)
            deviation = (val - med) / scale
            imp = float(self.tree_importances.get(fname, 0.0))

            # Attribution is scaled deviation weighted by tree importance
            attr = deviation * imp

            attributions.append({
                "feature": fname,
                "attribution": round(float(attr), 4),
                "abs_attribution": abs(float(attr)),
                "feature_value": val,
                "baseline_median": med,
                "impact_direction": "increases_anomaly_risk" if attr > 0 else "decreases_anomaly_risk",
            })

        attributions.sort(key=lambda x: x["abs_attribution"], reverse=True)
        top_features = attributions[:top_n]

        note = (
            "Using tree-weighted robust deviation surrogate attribution. "
            + ("SHAP not installed in environment." if not SHAP_AVAILABLE else "SHAP explainer fallback active.")
        )

        return {
            "method": "tree_feature_attribution_surrogate",
            "shap_available": SHAP_AVAILABLE,
            "status": "fallback_active",
            "note": note,
            "predicted_class": str(predicted_class) if predicted_class is not None else "Unknown",
            "top_features": top_features,
            "summary": _build_explanation_summary(top_features, predicted_class),
        }


def _build_explanation_summary(
    top_features: List[Dict],
    predicted_class: Optional[str],
) -> str:
    """Generate human-readable summary of top contributing factors."""
    if not top_features:
        return "No feature attribution information available."

    top_names = [f"'{tf['feature']}' (val={tf['feature_value']:.2f}, impact={tf['attribution']:+.2f})" for tf in top_features[:3]]
    target_text = f" for '{predicted_class}'" if predicted_class else ""
    return f"Key features influencing prediction{target_text}: {', '.join(top_names)}."

