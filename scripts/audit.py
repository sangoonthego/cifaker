"""Local fixture and dataset directory audit CLI for ProofLens (Member 1 & Member 4).

Scans a target dataset root directory (such as `data/fixtures` locally or
`/kaggle/input/tiny-genimage` on Kaggle in Phase 0B), validates the manifest
contract, and prints/saves an audit summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prooflens.data.manifest import save_manifest_csv
from prooflens.data.scanner import scan_dataset_directory
from prooflens.data.validation import summarize_manifest_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a dataset directory and validate its manifest contract."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of the dataset or synthetic fixtures (e.g., data/fixtures).",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional path to write the scanned manifest CSV.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional path to write the JSON audit summary.",
    )
    args = parser.parse_args(argv)

    df = scan_dataset_directory(args.root, include_corrupted=True)
    summary = summarize_manifest_audit(df)

    if args.manifest_out is not None:
        save_manifest_csv(df, args.manifest_out)

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
