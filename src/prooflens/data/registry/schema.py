"""Canonical image record contract and Parquet serialization for ProofLens.

Defines the canonical dataset schema used across inventory, leakage grouping,
deterministic nested sampling, materialization, and published releases.

Logical field definitions:
- Required logical fields:
  image_id, source_id, source_revision, source_path, source_split, label,
  generator, source_generator_group, source_shard, imagenet_class,
  leakage_group_id, eligibility_status
- Optional metadata fields:
  width, height, image_format, file_size_bytes, sha256, pixel_hash,
  perceptual_hash, decode_status, exclusion_reason
- Release/shard assignment columns (added during planning & materialization):
  release_split, shard_id, shard_storage_key, shard_member_name,
  shard_offset_data, origin_release_id

Critical provenance rule:
- REAL = 0, FAKE = 1.
- For REAL images (`label == 0`), `generator` MUST NOT name a generative model
  (must be one of `{"none", "real", "natural"}`). The paired upstream subset
  folder (e.g., `biggan`, `sdv5`) is tracked separately in `source_generator_group`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LABEL_REAL: int = 0
LABEL_FAKE: int = 1
VALID_LABELS: frozenset[int] = frozenset({LABEL_REAL, LABEL_FAKE})
ALLOWED_REAL_GENERATOR_VALUES: frozenset[str] = frozenset({"none", "real", "natural"})
VALID_ELIGIBILITY_STATUSES: frozenset[str] = frozenset({"ELIGIBLE", "EXCLUDED", "PENDING"})

CANONICAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "image_id",
    "source_id",
    "source_revision",
    "source_path",
    "source_split",
    "label",
    "generator",
    "source_generator_group",
    "source_shard",
    "imagenet_class",
    "leakage_group_id",
    "eligibility_status",
)

CANONICAL_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "width",
    "height",
    "image_format",
    "file_size_bytes",
    "sha256",
    "pixel_hash",
    "perceptual_hash",
    "decode_status",
    "exclusion_reason",
)

RELEASE_SHARD_COLUMNS: tuple[str, ...] = (
    "release_split",
    "shard_id",
    "shard_storage_key",
    "shard_member_name",
    "shard_offset_data",
    "origin_release_id",
)

ALL_CANONICAL_COLUMNS: tuple[str, ...] = CANONICAL_REQUIRED_COLUMNS + CANONICAL_OPTIONAL_COLUMNS

# PyArrow schema for deterministic, type-safe Parquet persistence
CANONICAL_ARROW_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
        pa.field("label", pa.int8(), nullable=False),
        pa.field("generator", pa.string(), nullable=False),
        pa.field("source_generator_group", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=True),
        pa.field("imagenet_class", pa.string(), nullable=True),
        pa.field("leakage_group_id", pa.string(), nullable=False),
        pa.field("eligibility_status", pa.string(), nullable=False),
        pa.field("width", pa.int32(), nullable=True),
        pa.field("height", pa.int32(), nullable=True),
        pa.field("image_format", pa.string(), nullable=True),
        pa.field("file_size_bytes", pa.int64(), nullable=True),
        pa.field("sha256", pa.string(), nullable=True),
        pa.field("pixel_hash", pa.string(), nullable=True),
        pa.field("perceptual_hash", pa.string(), nullable=True),
        pa.field("decode_status", pa.string(), nullable=True),
        pa.field("exclusion_reason", pa.string(), nullable=True),
    ]
)

_IMAGENET_SYNSET_RE = re.compile(r"(n\d{8})", re.IGNORECASE)
_GENIMAGE_CLASS_PREFIX_RE = re.compile(r"^(\d{3})_[a-zA-Z0-9]+_\d+")


def validate_safe_relative_path(raw_path: str) -> str:
    """Normalize a relative source path to POSIX form and reject traversal or absolute paths."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Source path must be a non-empty string.")
    norm = raw_path.strip().replace("\\", "/")
    if norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        raise ValueError(f"Unsafe absolute path rejected: {raw_path!r}")
    if "\x00" in norm:
        raise ValueError(f"NUL byte in path rejected: {raw_path!r}")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"Unsafe path traversal ('..') rejected: {raw_path!r}")
    return "/".join(parts)


def make_canonical_image_id(source_id: str, source_revision: str, source_path: str) -> str:
    """Derive a stable, collision-resistant canonical image ID purely from source provenance."""
    safe_path = validate_safe_relative_path(source_path)
    payload = f"{source_id.strip()}|{source_revision.strip()}|{safe_path}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"pl_{digest}"


def extract_imagenet_class_from_path(source_path: str) -> str | None:
    """Extract ImageNet synset (`n01440764`) or 3-digit class prefix (`000`) when reliably present."""
    filename = Path(source_path).name
    synset_match = _IMAGENET_SYNSET_RE.search(filename)
    if synset_match:
        return synset_match.group(1).lower()
    prefix_match = _GENIMAGE_CLASS_PREFIX_RE.match(filename)
    if prefix_match:
        return f"cls_{prefix_match.group(1)}"
    return None


@dataclass(frozen=True)
class CanonicalImageRecord:
    """Canonical record representing one discovered or materialized dataset image."""

    image_id: str
    source_id: str
    source_revision: str
    source_path: str
    source_split: str
    label: int
    generator: str
    source_generator_group: str
    source_shard: str | None = None
    imagenet_class: str | None = None
    leakage_group_id: str = ""
    eligibility_status: str = "ELIGIBLE"
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    file_size_bytes: int | None = None
    sha256: str | None = None
    pixel_hash: str | None = None
    perceptual_hash: str | None = None
    decode_status: str | None = None
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("CanonicalImageRecord.image_id cannot be empty.")
        validate_safe_relative_path(self.source_path)
        if self.label not in VALID_LABELS:
            raise ValueError(f"Invalid binary label {self.label!r}; expected REAL=0 or FAKE=1.")
        if self.label == LABEL_REAL and self.generator.lower() not in ALLOWED_REAL_GENERATOR_VALUES:
            raise ValueError(
                f"REAL image (label=0) cannot have generator={self.generator!r}; "
                f"expected one of {sorted(ALLOWED_REAL_GENERATOR_VALUES)}. "
                "Track paired GenImage folder in 'source_generator_group'."
            )
        if self.label == LABEL_FAKE and (
            not self.generator.strip() or self.generator.lower() in ALLOWED_REAL_GENERATOR_VALUES
        ):
            raise ValueError(
                f"FAKE image (label=1) must identify a generative model, got generator={self.generator!r}."
            )
        if self.eligibility_status not in VALID_ELIGIBILITY_STATUSES:
            raise ValueError(
                f"Invalid eligibility_status={self.eligibility_status!r}; "
                f"expected one of {sorted(VALID_ELIGIBILITY_STATUSES)}."
            )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if not row["leakage_group_id"]:
            row["leakage_group_id"] = f"lg_{self.image_id}"
        return row


def validate_canonical_dataframe(
    df: pd.DataFrame,
    require_release_columns: bool = False,
) -> pd.DataFrame:
    """Validate a canonical metadata DataFrame against the ProofLens data contract.

    Checks:
    - Required logical columns exist.
    - No duplicate `image_id`.
    - Safe relative `source_path` with no absolute paths or `..`.
    - Strict binary `label` in `{0, 1}`.
    - REAL images do not claim a generative model in `generator`.
    - FAKE images specify a non-empty generative model in `generator`.
    - Valid `eligibility_status`.
    """
    required = list(CANONICAL_REQUIRED_COLUMNS)
    if require_release_columns:
        required.extend(RELEASE_SHARD_COLUMNS)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Canonical table missing required columns: {missing}")

    if df.empty:
        return df.copy()

    non_null_required = (
        "image_id",
        "source_id",
        "source_revision",
        "source_path",
        "source_split",
        "label",
        "generator",
        "source_generator_group",
        "leakage_group_id",
        "eligibility_status",
    )
    for col in non_null_required:
        if df[col].isna().any():
            raise ValueError(f"Canonical table has null values in required column '{col}'.")
        if col != "label" and (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"Canonical table has empty string in required column '{col}'.")

    # Duplicate image_id check
    dup_mask = df["image_id"].duplicated()
    if dup_mask.any():
        dups = df.loc[dup_mask, "image_id"].head(5).tolist()
        raise ValueError(f"Duplicate canonical image_id detected: {dups}")

    # Binary label check
    try:
        labels = pd.to_numeric(df["label"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Canonical 'label' column must be numeric binary {0, 1}.") from exc
    if not labels.isin(VALID_LABELS).all():
        bad = df.loc[~labels.isin(VALID_LABELS), "label"].unique().tolist()
        raise ValueError(f"Invalid canonical binary labels: {bad}")

    # REAL vs FAKE generator provenance hygiene
    gen_lower = df["generator"].astype(str).str.strip().str.lower()
    bad_real = (labels == LABEL_REAL) & (~gen_lower.isin(ALLOWED_REAL_GENERATOR_VALUES))
    if bad_real.any():
        samples = df.loc[bad_real, ["image_id", "generator"]].head(5).to_dict(orient="records")
        raise ValueError(
            "REAL images must not have generative model names in 'generator' "
            f"(use 'source_generator_group' instead). Examples: {samples}"
        )

    bad_fake = (labels == LABEL_FAKE) & (gen_lower.isin(ALLOWED_REAL_GENERATOR_VALUES))
    if bad_fake.any():
        samples = df.loc[bad_fake, ["image_id", "generator"]].head(5).to_dict(orient="records")
        raise ValueError(
            f"FAKE images must specify a generative model in 'generator'. Examples: {samples}"
        )

    # Eligibility check
    bad_elig = ~df["eligibility_status"].isin(VALID_ELIGIBILITY_STATUSES)
    if bad_elig.any():
        vals = df.loc[bad_elig, "eligibility_status"].unique().tolist()
        raise ValueError(f"Invalid eligibility_status values: {vals}")

    # Safe relative source paths
    normalized_paths: list[str] = []
    for raw_p in df["source_path"].astype(str):
        normalized_paths.append(validate_safe_relative_path(raw_p))

    out = df.copy()
    out["label"] = labels.astype("int8")
    out["source_path"] = normalized_paths
    for opt in CANONICAL_OPTIONAL_COLUMNS:
        if opt not in out.columns:
            out[opt] = None

    return out


def dataframe_to_canonical_arrow(df: pd.DataFrame) -> pa.Table:
    """Convert a validated DataFrame into a typed PyArrow Table preserving extra release columns."""
    fields = list(CANONICAL_ARROW_SCHEMA)
    known_names = {f.name for f in fields}
    extra_cols = [c for c in df.columns if c not in known_names]

    table_cols: dict[str, Any] = {}
    for field_spec in fields:
        if field_spec.name in df.columns:
            series = df[field_spec.name]
        else:
            series = pd.Series([None] * len(df))
        if pa.types.is_integer(field_spec.type):
            table_cols[field_spec.name] = pa.array(
                pd.to_numeric(series, errors="coerce").astype("Int64"),
                type=field_spec.type,
            )
        else:
            table_cols[field_spec.name] = pa.array(
                series.where(series.notna(), None).map(lambda x: str(x) if x is not None else None),
                type=field_spec.type,
            )

    extra_fields: list[pa.Field] = []
    for col in extra_cols:
        arr = pa.array(df[col].where(df[col].notna(), None))
        table_cols[col] = arr
        extra_fields.append(pa.field(col, arr.type, nullable=True))

    schema = pa.schema(fields + extra_fields)
    return pa.Table.from_pydict(table_cols, schema=schema)


def write_canonical_parquet(
    df: pd.DataFrame,
    output_path: Path | str,
    require_release_columns: bool = False,
) -> Path:
    """Validate and write a canonical metadata table deterministically to Parquet."""
    validated = validate_canonical_dataframe(df, require_release_columns=require_release_columns)
    if not validated.empty:
        validated = validated.sort_values(by=["image_id"], kind="mergesort").reset_index(drop=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = dataframe_to_canonical_arrow(validated)
    pq.write_table(table, path, compression="snappy")
    return path


def read_canonical_parquet(
    parquet_path: Path | str,
    columns: list[str] | None = None,
    validate: bool = True,
    require_release_columns: bool = False,
) -> pd.DataFrame:
    """Read a canonical Parquet file and optionally validate its schema."""
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical Parquet file not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    if validate and columns is None:
        return validate_canonical_dataframe(df, require_release_columns=require_release_columns)
    return df
