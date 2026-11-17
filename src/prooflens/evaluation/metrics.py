"""Evaluation contract and classification metrics for ProofLens (Member 3).

Prediction Contract Schema:
- `image_id`: Unique image identifier
- `model`: Model architecture name
- `generator`: Source generator identifier
- `y_true`: Ground-truth binary label (`0` for REAL, `1` for FAKE)
- `y_score`: Sigmoid probability score for FAKE in `[0.0, 1.0]`
- `y_pred`: Binary prediction in `{0, 1}` using a documented threshold
- `run_id`: Experiment run identifier

Metric definitions:
- `accuracy`: `(TP + TN) / (TP + TN + FP + FN)`
- `precision`: `TP / (TP + FP)`; returns `None` (`np.nan` in numeric tables) if `TP + FP == 0`
- `recall`: `TP / (TP + FN)`; returns `None` if `TP + FN == 0`
- `f1`: `2 * TP / (2 * TP + FP + FN)`; returns `None` if `2 * TP + FP + FN == 0`
- `balanced_accuracy`: `0.5 * (TPR + TNR)`; returns `None` if only one ground-truth class is present
- `auroc`: Area Under the ROC Curve; returns `None` if only one ground-truth class is present
- `average_precision`: Area Under Precision-Recall Curve; returns `None` if only one ground-truth class is present
- `confusion_matrix`: Dictionary with `tn`, `fp`, `fn`, `tp`

Undefined metrics are explicitly represented as `None` (never fabricated zeros).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "image_id",
    "model",
    "generator",
    "y_true",
    "y_score",
    "y_pred",
    "run_id",
)


def validate_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the ProofLens prediction output contract.

    Raises:
        ValueError: On empty input, missing columns, missing/NaN values, duplicate `image_id`,
            invalid binary labels/predictions, or out-of-range `y_score` values.
    """
    missing_cols = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Prediction table is missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Prediction table is empty.")

    for col in REQUIRED_PREDICTION_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"Prediction table contains missing/NaN values in column '{col}'.")

    for col in ("image_id", "model", "generator", "run_id"):
        if (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"Prediction table contains empty strings in column '{col}'.")

    if df["image_id"].duplicated().any():
        dups = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Prediction table contains duplicate image_id entries: {dups[:5]}")

    try:
        y_true = pd.to_numeric(df["y_true"], errors="raise")
        y_pred = pd.to_numeric(df["y_pred"], errors="raise")
        y_score = pd.to_numeric(df["y_score"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Prediction columns y_true, y_pred, y_score must be numeric.") from exc

    if not y_true.isin({0, 1}).all():
        raise ValueError("Prediction column 'y_true' must contain only binary values {0, 1}.")

    if not y_pred.isin({0, 1}).all():
        raise ValueError("Prediction column 'y_pred' must contain only binary values {0, 1}.")

    if not np.isfinite(y_score).all() or ((y_score < 0.0) | (y_score > 1.0)).any():
        raise ValueError("Prediction column 'y_score' must contain finite probabilities in [0, 1].")

    out = df.copy()
    out["y_true"] = y_true.astype(int)
    out["y_pred"] = y_pred.astype(int)
    out["y_score"] = y_score.astype(float)
    return out


def compute_binary_metrics(predictions_df: pd.DataFrame) -> dict[str, Any]:
    """Compute binary classification metrics from a validated prediction DataFrame.

    Never fabricates zeros when a metric is mathematically undefined (e.g. single-class subset
    or zero predicted/actual positives); returns `None` for undefined metrics.
    """
    validated = validate_prediction_dataframe(predictions_df)
    y_true = validated["y_true"].to_numpy(dtype=int)
    y_pred = validated["y_pred"].to_numpy(dtype=int)
    y_score = validated["y_score"].to_numpy(dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    total = tn + fp + fn + tp

    accuracy: float | None = float((tp + tn) / total) if total > 0 else None
    precision: float | None = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall: float | None = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity: float | None = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1: float | None = float(2.0 * precision * recall / (precision + recall))
    elif (tp + fp) > 0 and (tp + fn) > 0 and tp == 0:
        # Both positive predictions and positive ground-truth exist, but TP == 0 -> F1 is defined as 0.0
        f1 = 0.0
    else:
        f1 = None

    has_both_classes = (tp + fn) > 0 and (tn + fp) > 0
    if has_both_classes and recall is not None and specificity is not None:
        balanced_accuracy: float | None = float(0.5 * (recall + specificity))
        auroc: float | None = float(roc_auc_score(y_true, y_score))
        avg_precision: float | None = float(average_precision_score(y_true, y_score))
    else:
        balanced_accuracy = None
        auroc = None
        avg_precision = None

    return {
        "n_samples": int(total),
        "n_real": int(tn + fp),
        "n_fake": int(tp + fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "average_precision": avg_precision,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
