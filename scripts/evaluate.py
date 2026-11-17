"""Prediction evaluation and subgroup reporting CLI for ProofLens (Member 3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prooflens.evaluation.metrics import compute_binary_metrics
from prooflens.evaluation.subgroups import compute_subgroup_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a ProofLens prediction CSV table and compute overall & subgroup metrics."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to a prediction CSV file conforming to the Prediction Contract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save evaluation JSON output.",
    )
    args = parser.parse_args(argv)

    if not args.predictions.is_file():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    preds_df = pd.read_csv(args.predictions)
    overall = compute_binary_metrics(preds_df)
    by_generator = compute_subgroup_metrics(preds_df, group_by="generator").to_dict(
        orient="records"
    )

    report = {
        "predictions_file": str(args.predictions),
        "overall_metrics": overall,
        "generator_subgroups": by_generator,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
