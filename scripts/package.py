"""Source packaging utility for uploading ProofLens code to Kaggle Notebooks (Member 4).

Creates a clean, portable ZIP archive containing `src/`, `configs/`, `scripts/`,
`pyproject.toml`, and `README.md` while strictly excluding raw datasets, synthetic
fixtures, virtual environments, checkpoints, artifacts, and secrets.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

INCLUDED_PATHS: tuple[str, ...] = (
    "src",
    "configs",
    "scripts",
    "pyproject.toml",
    "README.md",
)

EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".pytest_cache",
        ".ruff_cache",
        "data",
        "artifacts",
        "outputs",
        "checkpoints",
    }
)


def package_repository(
    repo_root: Path | str = Path("."),
    output_zip: Path | str = Path("artifacts/prooflens_code.zip"),
) -> Path:
    """Package ProofLens source code and configurations into a clean ZIP archive."""
    root = Path(repo_root).resolve()
    out_path = Path(output_zip)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_item in INCLUDED_PATHS:
            target = root / rel_item
            if not target.exists():
                continue
            if target.is_file():
                zf.write(target, arcname=rel_item)
            else:
                for path in sorted(target.rglob("*")):
                    if path.is_dir():
                        continue
                    if any(part in EXCLUDED_DIRS for part in path.parts):
                        continue
                    if path.suffix in {".pyc", ".pyo", ".pt", ".pth", ".ckpt"}:
                        continue
                    arcname = path.relative_to(root).as_posix()
                    zf.write(path, arcname=arcname)

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package ProofLens code and configs into a lightweight ZIP for Kaggle."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prooflens_code.zip"),
        help="Destination ZIP path (default: artifacts/prooflens_code.zip).",
    )
    args = parser.parse_args(argv)

    archive_path = package_repository(Path("."), args.output)
    print(f"[ProofLens] Packaged source code to: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
