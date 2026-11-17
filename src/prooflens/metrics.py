"""Evaluation metrics placeholder for ProofLens (Deferred to Phase 1+).

Planned metrics for later phases:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC (Average Precision)
- Expected Calibration Error (ECE)

NOTE: No fabricated or placeholder metric values are generated in Phase 0.
"""

from __future__ import annotations

PLANNED_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "ece",
)


def compute_classification_metrics(*args: object, **kwargs: object) -> dict[str, float]:
    """Placeholder for Phase 1+ classification metric calculation."""
    raise NotImplementedError("Metric computation on model predictions is deferred to Phase 1.")
