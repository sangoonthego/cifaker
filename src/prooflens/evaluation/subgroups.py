"""Subgroup metric aggregation for ProofLens (Member 3).

Supports stratified evaluation across generators, classes, or custom metadata columns,
handling single-class subgroups (such as generator-specific FAKE subsets) cleanly
without fabricating undefined metrics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from prooflens.evaluation.metrics import compute_binary_metrics, validate_prediction_dataframe


def compute_subgroup_metrics(
    predictions_df: pd.DataFrame,
    group_by: str | list[str] = "generator",
) -> pd.DataFrame:
    """Aggregate classification metrics across subgroups (e.g., per `generator` or `model`).

    Args:
        predictions_df: DataFrame conforming to the Prediction Contract.
        group_by: Column name or list of column names to group by.

    Returns:
        DataFrame with one row per subgroup and columns for sample counts, metrics,
        and confusion matrix components (`tn`, `fp`, `fn`, `tp`).
    """
    validated = validate_prediction_dataframe(predictions_df)
    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)

    for col in group_cols:
        if col not in validated.columns:
            raise ValueError(f"Grouping column '{col}' not found in prediction DataFrame.")

    rows: list[dict[str, Any]] = []
    for group_keys, sub_df in validated.groupby(group_cols, sort=True):
        key_tuple = (group_keys,) if not isinstance(group_keys, tuple) else group_keys
        row_dict: dict[str, Any] = dict(zip(group_cols, key_tuple, strict=True))

        metrics = compute_binary_metrics(sub_df)
        cm = metrics.pop("confusion_matrix")
        row_dict.update(metrics)
        row_dict.update(cm)
        rows.append(row_dict)

    return pd.DataFrame(rows)
