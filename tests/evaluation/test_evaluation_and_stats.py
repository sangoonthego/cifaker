"""Unit tests for ProofLens evaluation contract, edge cases, subgroups, and statistics (Member 3)."""

from __future__ import annotations

import pandas as pd
import pytest

from prooflens.evaluation.metrics import compute_binary_metrics, validate_prediction_dataframe
from prooflens.evaluation.statistics import (
    difference_in_differences_contrast,
    paired_bootstrap_metric_diff,
)
from prooflens.evaluation.subgroups import compute_subgroup_metrics


def _make_synthetic_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "id_1",
                "model": "resnet50",
                "generator": "gen_a",
                "y_true": 0,
                "y_score": 0.1,
                "y_pred": 0,
                "run_id": "r1",
            },
            {
                "image_id": "id_2",
                "model": "resnet50",
                "generator": "gen_a",
                "y_true": 1,
                "y_score": 0.9,
                "y_pred": 1,
                "run_id": "r1",
            },
            {
                "image_id": "id_3",
                "model": "resnet50",
                "generator": "gen_b",
                "y_true": 0,
                "y_score": 0.6,
                "y_pred": 1,
                "run_id": "r1",
            },
            {
                "image_id": "id_4",
                "model": "resnet50",
                "generator": "gen_b",
                "y_true": 1,
                "y_score": 0.8,
                "y_pred": 1,
                "run_id": "r1",
            },
        ]
    )


def test_metrics_computation_and_confusion_matrix() -> None:
    df = _make_synthetic_predictions()
    metrics = compute_binary_metrics(df)

    assert metrics["n_samples"] == 4
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_single_class_subset_does_not_fabricate_zeros() -> None:
    df = _make_synthetic_predictions()
    only_real = df[df["y_true"] == 0].copy()
    only_real["y_pred"] = 0
    only_real["y_score"] = 0.1

    res = compute_binary_metrics(only_real)
    assert res["accuracy"] == pytest.approx(1.0)
    # Positive class (FAKE=1) has 0 true samples and 0 predicted samples -> undefined metrics must be None
    assert res["precision"] is None
    assert res["recall"] is None
    assert res["f1"] is None
    assert res["balanced_accuracy"] is None
    assert res["auroc"] is None
    assert res["average_precision"] is None


def test_prediction_validation_edge_cases() -> None:
    df = _make_synthetic_predictions()

    # Empty DataFrame
    with pytest.raises(ValueError, match="empty"):
        validate_prediction_dataframe(df.iloc[0:0])

    # Duplicate image_id
    dup = df.copy()
    dup.loc[1, "image_id"] = "id_1"
    with pytest.raises(ValueError, match="duplicate image_id"):
        validate_prediction_dataframe(dup)

    # Invalid label
    bad_y = df.copy()
    bad_y.loc[0, "y_true"] = 3
    with pytest.raises(ValueError, match="binary values"):
        validate_prediction_dataframe(bad_y)

    # Missing value
    missing_val = df.copy()
    missing_val.loc[0, "y_score"] = None
    with pytest.raises(ValueError, match="missing/NaN"):
        validate_prediction_dataframe(missing_val)


def test_subgroup_aggregation_and_statistical_utilities() -> None:
    preds_a = _make_synthetic_predictions()
    sub_df = compute_subgroup_metrics(preds_a, group_by="generator")
    assert len(sub_df) == 2
    assert set(sub_df["generator"]) == {"gen_a", "gen_b"}

    # Paired bootstrap comparison against a weaker model B
    preds_b = preds_a.copy()
    preds_b["model"] = "densenet121"
    preds_b["y_pred"] = [1, 1, 1, 0]
    preds_b["y_score"] = [0.7, 0.6, 0.8, 0.2]

    boot = paired_bootstrap_metric_diff(
        preds_a, preds_b, metric="accuracy", n_bootstraps=100, seed=42
    )
    assert boot["observed_a"] == pytest.approx(0.75)
    assert boot["observed_b"] == pytest.approx(0.25)
    assert boot["observed_diff"] == pytest.approx(0.50)
    assert boot["ci_lower"] is not None
    assert boot["ci_upper"] is not None
    assert boot["n_valid_bootstraps"] == 100

    # Difference-in-Differences 2x2 factorial contrast
    factorial_df = pd.DataFrame(
        [
            {"aug_jpeg": 0, "freq_branch": 0, "f1": 0.70},
            {"aug_jpeg": 0, "freq_branch": 1, "f1": 0.74},
            {"aug_jpeg": 1, "freq_branch": 0, "f1": 0.75},
            {"aug_jpeg": 1, "freq_branch": 1, "f1": 0.85},
        ]
    )
    did = difference_in_differences_contrast(
        factorial_df,
        factor_a_col="aug_jpeg",
        factor_b_col="freq_branch",
        metric_col="f1",
    )
    # Effect of B when A=1 is (0.85 - 0.75) = 0.10; when A=0 is (0.74 - 0.70) = 0.04 -> DiD = +0.06
    assert did["interaction_did"] == pytest.approx(0.06)
    assert did["main_effect_a"] == pytest.approx(0.08)
    assert did["main_effect_b"] == pytest.approx(0.07)
