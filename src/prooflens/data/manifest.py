"""Dataset manifest contract and CSV serialization for ProofLens.

Defines the canonical manifest schema required across all four team members:
Required columns:
- image_id
- relative_path
- generator
- label
- source_split

Optional metadata columns:
- width
- height
- image_format
- file_size_bytes
- sha256
- decode_status
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_REAL: int = 0
LABEL_FAKE: int = 1
VALID_BINARY_LABELS: frozenset[int] = frozenset({LABEL_REAL, LABEL_FAKE})

REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "relative_path",
    "generator",
    "label",
    "source_split",
)

OPTIONAL_MANIFEST_COLUMNS: tuple[str, ...] = (
    "width",
    "height",
    "image_format",
    "file_size_bytes",
    "sha256",
    "decode_status",
)

ALL_MANIFEST_COLUMNS: tuple[str, ...] = REQUIRED_MANIFEST_COLUMNS + OPTIONAL_MANIFEST_COLUMNS


@dataclass(frozen=True)
class ManifestRow:
    """Single validated record in a ProofLens dataset manifest."""

    image_id: str
    relative_path: str
    generator: str
    label: int
    source_split: str
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    file_size_bytes: int | None = None
    sha256: str | None = None
    decode_status: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest row to a plain dictionary."""
        return asdict(self)


def validate_manifest_dataframe(
    df: pd.DataFrame,
    allow_unknown_generator: bool = True,
) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the ProofLens dataset manifest contract.

    Checks:
    1. All required columns (`image_id`, `relative_path`, `generator`, `label`, `source_split`) exist.
    2. No required fields contain missing/null/empty values.
    3. `image_id` values are unique (no duplicate image IDs).
    4. `relative_path` values are relative POSIX paths (not absolute paths or parent `..` traversals).
    5. `label` values are strictly binary integers in `{0, 1}` (REAL=0, FAKE=1).
    6. `generator` values are non-empty strings (and if `allow_unknown_generator=False`, not `'unknown'`).

    Returns:
        A deterministically sorted copy of the validated DataFrame (sorted by `relative_path`, `image_id`).
    """
    missing_cols = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")

    if df.empty:
        return df.copy()

    # Check null or empty required strings
    for col in ("image_id", "relative_path", "generator", "source_split"):
        if df[col].isna().any():
            raise ValueError(f"Manifest contains null values in required column '{col}'.")
        str_vals = df[col].astype(str).str.strip()
        if (str_vals == "").any():
            raise ValueError(f"Manifest contains empty strings in required column '{col}'.")

    # Reject duplicate image_id
    duplicated_ids = df[df["image_id"].duplicated()]["image_id"].tolist()
    if duplicated_ids:
        raise ValueError(f"Manifest contains duplicate image_id values: {duplicated_ids[:5]}")

    # Validate binary labels
    if df["label"].isna().any():
        raise ValueError("Manifest contains null values in 'label'.")

    try:
        numeric_labels = pd.to_numeric(df["label"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Manifest 'label' column contains non-numeric values.") from exc

    if not ((numeric_labels == LABEL_REAL) | (numeric_labels == LABEL_FAKE)).all():
        invalid_vals = df.loc[~numeric_labels.isin(VALID_BINARY_LABELS), "label"].unique().tolist()
        raise ValueError(
            f"Manifest contains invalid binary labels {invalid_vals}. Expected REAL=0 or FAKE=1."
        )

    # Validate relative paths
    for raw_p in df["relative_path"].astype(str):
        norm_p = raw_p.replace("\\", "/")
        p_obj = Path(norm_p)
        if p_obj.is_absolute() or norm_p.startswith("/") or (len(norm_p) >= 2 and norm_p[1] == ":"):
            raise ValueError(
                f"Manifest 'relative_path' must be relative, got absolute path: {raw_p}"
            )
        if ".." in p_obj.parts:
            raise ValueError(
                f"Manifest 'relative_path' must not traverse parent directories: {raw_p}"
            )

    # Handle unknown generator policy
    if not allow_unknown_generator:
        unknown_mask = df["generator"].astype(str).str.lower() == "unknown"
        if unknown_mask.any():
            raise ValueError("Manifest contains 'unknown' generator entries when disallowed.")

    validated = df.copy()
    validated["label"] = numeric_labels.astype(int)
    validated["relative_path"] = (
        validated["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    validated["generator"] = validated["generator"].astype(str).str.strip()
    validated["source_split"] = validated["source_split"].astype(str).str.strip()

    # Deterministic row ordering
    validated = validated.sort_values(
        by=["relative_path", "image_id"], kind="mergesort"
    ).reset_index(drop=True)
    return validated


def save_manifest_csv(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Validate and save a dataset manifest DataFrame deterministically to CSV."""
    validated = validate_manifest_dataframe(df)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)
    return path


def load_manifest_csv(
    manifest_path: Path | str,
    allow_unknown_generator: bool = True,
) -> pd.DataFrame:
    """Load and validate a dataset manifest from a CSV file."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    df = pd.read_csv(path)
    return validate_manifest_dataframe(df, allow_unknown_generator=allow_unknown_generator)
