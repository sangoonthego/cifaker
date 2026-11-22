"""Canonical registry schema, streaming inventory, and duplicate detection."""

from __future__ import annotations

from prooflens.data.registry.duplicates import (
    DuplicateAuditReport,
    assign_leakage_groups,
    find_perceptual_near_duplicate_candidates,
    hamming_distance_hex64,
)
from prooflens.data.registry.inventory import InventoryResult, build_source_inventory
from prooflens.data.registry.schema import (
    ALL_CANONICAL_COLUMNS,
    ALLOWED_REAL_GENERATOR_VALUES,
    CANONICAL_ARROW_SCHEMA,
    CANONICAL_OPTIONAL_COLUMNS,
    CANONICAL_REQUIRED_COLUMNS,
    LABEL_FAKE,
    LABEL_REAL,
    RELEASE_SHARD_COLUMNS,
    CanonicalImageRecord,
    dataframe_to_canonical_arrow,
    extract_imagenet_class_from_path,
    make_canonical_image_id,
    read_canonical_parquet,
    validate_canonical_dataframe,
    validate_safe_relative_path,
    write_canonical_parquet,
)

__all__ = [
    "ALL_CANONICAL_COLUMNS",
    "ALLOWED_REAL_GENERATOR_VALUES",
    "CANONICAL_ARROW_SCHEMA",
    "CANONICAL_OPTIONAL_COLUMNS",
    "CANONICAL_REQUIRED_COLUMNS",
    "CanonicalImageRecord",
    "DuplicateAuditReport",
    "InventoryResult",
    "LABEL_FAKE",
    "LABEL_REAL",
    "RELEASE_SHARD_COLUMNS",
    "assign_leakage_groups",
    "build_source_inventory",
    "dataframe_to_canonical_arrow",
    "extract_imagenet_class_from_path",
    "find_perceptual_near_duplicate_candidates",
    "hamming_distance_hex64",
    "make_canonical_image_id",
    "read_canonical_parquet",
    "validate_canonical_dataframe",
    "validate_safe_relative_path",
    "write_canonical_parquet",
]
