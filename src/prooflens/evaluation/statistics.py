"""Statistical validation utilities for ProofLens (Member 3).

Implements:
1. Paired bootstrap confidence intervals and hypothesis testing comparing two models
   on the exact same evaluation images (`image_id` aligned).
2. Difference-in-differences (2x2 factorial interaction contrast) for studying
   feature interactions and augmentation effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prooflens.evaluation.metrics import compute_binary_metrics, validate_prediction_dataframe


def paired_bootstrap_metric_diff(
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    metric: str = "accuracy",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute a paired bootstrap confidence interval for `metric(model_A) - metric(model_B)`.

    Both prediction tables must contain the exact same set of `image_id` rows so that
    resampling preserves image-level pairing.
    """
    if n_bootstraps <= 0:
        raise ValueError(f"n_bootstraps must be positive, got {n_bootstraps}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    val_a = validate_prediction_dataframe(preds_a).sort_values("image_id").reset_index(drop=True)
    val_b = validate_prediction_dataframe(preds_b).sort_values("image_id").reset_index(drop=True)

    if (
        len(val_a) != len(val_b)
        or not (val_a["image_id"].to_numpy() == val_b["image_id"].to_numpy()).all()
    ):
        raise ValueError(
            "Paired bootstrap requires identical image_id sets across both prediction tables."
        )

    if not (val_a["y_true"].to_numpy() == val_b["y_true"].to_numpy()).all():
        raise ValueError(
            "Ground-truth labels 'y_true' do not match between paired prediction tables."
        )

    base_metrics_a = compute_binary_metrics(val_a)
    base_metrics_b = compute_binary_metrics(val_b)

    if metric not in base_metrics_a:
        raise ValueError(f"Unsupported metric '{metric}'. Available: {list(base_metrics_a.keys())}")

    obs_a = base_metrics_a[metric]
    obs_b = base_metrics_b[metric]
    if obs_a is None or obs_b is None:
        return {
            "metric": metric,
            "observed_a": obs_a,
            "observed_b": obs_b,
            "observed_diff": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value_two_sided": None,
            "n_valid_bootstraps": 0,
        }

    observed_diff = float(obs_a - obs_b)
    rng = np.random.default_rng(seed)
    n_samples = len(val_a)

    diffs: list[float] = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=n_samples)
        sample_a = val_a.iloc[indices].copy()
        sample_b = val_b.iloc[indices].copy()
        # Assign unique temporary image_ids for bootstrap resamples with replacement
        temp_ids = [f"boot_{i}" for i in range(n_samples)]
        sample_a["image_id"] = temp_ids
        sample_b["image_id"] = temp_ids

        m_a = compute_binary_metrics(sample_a)[metric]
        m_b = compute_binary_metrics(sample_b)[metric]
        if m_a is not None and m_b is not None:
            diffs.append(float(m_a - m_b))

    if not diffs:
        return {
            "metric": metric,
            "observed_a": float(obs_a),
            "observed_b": float(obs_b),
            "observed_diff": observed_diff,
            "ci_lower": None,
            "ci_upper": None,
            "p_value_two_sided": None,
            "n_valid_bootstraps": 0,
        }

    diffs_arr = np.asarray(diffs, dtype=float)
    alpha = 1.0 - confidence_level
    ci_lower = float(np.quantile(diffs_arr, alpha / 2.0))
    ci_upper = float(np.quantile(diffs_arr, 1.0 - alpha / 2.0))

    # Two-sided empirical bootstrap p-value under null hypothesis diff == 0
    prop_le_zero = float(np.mean(diffs_arr <= 0.0))
    prop_ge_zero = float(np.mean(diffs_arr >= 0.0))
    p_val = float(min(1.0, 2.0 * min(prop_le_zero, prop_ge_zero)))

    return {
        "metric": metric,
        "observed_a": float(obs_a),
        "observed_b": float(obs_b),
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_two_sided": p_val,
        "n_valid_bootstraps": len(diffs),
    }


def difference_in_differences_contrast(
    outcomes_df: pd.DataFrame,
    factor_a_col: str,
    factor_b_col: str,
    metric_col: str,
) -> dict[str, float]:
    """Compute a 2x2 factorial Difference-in-Differences (DiD) interaction contrast.

    Given binary factor columns `factor_a_col` in `{0, 1}` and `factor_b_col` in `{0, 1}`:
        DiD = (E[Y | A=1, B=1] - E[Y | A=1, B=0]) - (E[Y | A=0, B=1] - E[Y | A=0, B=0])

    Also computes the main effects of Factor A and Factor B.
    """
    for col in (factor_a_col, factor_b_col, metric_col):
        if col not in outcomes_df.columns:
            raise ValueError(f"Column '{col}' not found in outcomes DataFrame.")

    if outcomes_df.empty:
        raise ValueError("outcomes_df is empty.")

    cell_means: dict[tuple[int, int], float] = {}
    for a_val in (0, 1):
        for b_val in (0, 1):
            mask = (outcomes_df[factor_a_col] == a_val) & (outcomes_df[factor_b_col] == b_val)
            cell_vals = outcomes_df.loc[mask, metric_col].dropna().astype(float)
            if cell_vals.empty:
                raise ValueError(
                    f"Missing factorial cell ({factor_a_col}={a_val}, {factor_b_col}={b_val})."
                )
            cell_means[(a_val, b_val)] = float(cell_vals.mean())

    y00 = cell_means[(0, 0)]
    y01 = cell_means[(0, 1)]
    y10 = cell_means[(1, 0)]
    y11 = cell_means[(1, 1)]

    effect_b_at_a1 = y11 - y10
    effect_b_at_a0 = y01 - y00
    did_interaction = effect_b_at_a1 - effect_b_at_a0

    main_effect_a = 0.5 * ((y11 - y01) + (y10 - y00))
    main_effect_b = 0.5 * ((y11 - y10) + (y01 - y00))

    return {
        "mean_a0_b0": y00,
        "mean_a0_b1": y01,
        "mean_a1_b0": y10,
        "mean_a1_b1": y11,
        "main_effect_a": float(main_effect_a),
        "main_effect_b": float(main_effect_b),
        "interaction_did": float(did_interaction),
    }
