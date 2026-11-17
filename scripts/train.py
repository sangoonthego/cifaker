"""CPU smoke training and experiment entrypoint CLI for ProofLens (Member 2 & Member 4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prooflens.experiments.runner import (
    execute_smoke_experiment,
    load_and_validate_experiment_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ProofLens CPU smoke training from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML configuration file (e.g., configs/smoke.yaml).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional override for dataset.root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional override for dataset.manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output.directory.",
    )
    args = parser.parse_args(argv)

    cfg = load_and_validate_experiment_config(
        config_path=args.config,
        dataset_root_override=args.dataset_root,
        manifest_override=args.manifest,
        output_dir_override=args.output_dir,
    )

    summary = execute_smoke_experiment(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
