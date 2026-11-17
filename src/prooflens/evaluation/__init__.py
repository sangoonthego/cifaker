"""Evaluation metrics, subgroup analysis, and statistics package for ProofLens (Member 3)."""

from __future__ import annotations

from prooflens.evaluation.metrics import (
    REQUIRED_PREDICTION_COLUMNS,
    compute_binary_metrics,
    validate_prediction_dataframe,
)
from prooflens.evaluation.statistics import (
    difference_in_differences_contrast,
    paired_bootstrap_metric_diff,
)
from prooflens.evaluation.subgroups import compute_subgroup_metrics

__all__ = [
    "REQUIRED_PREDICTION_COLUMNS",
    "compute_binary_metrics",
    "compute_subgroup_metrics",
    "difference_in_differences_contrast",
    "paired_bootstrap_metric_diff",
    "validate_prediction_dataframe",
]
