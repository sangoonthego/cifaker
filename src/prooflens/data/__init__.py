"""Dataset, manifest, scanner, and validation package for ProofLens (Member 1)."""

from __future__ import annotations

from prooflens.data.dataset import ProofLensDataset, build_image_transforms, create_dataloader
from prooflens.data.manifest import (
    ALL_MANIFEST_COLUMNS,
    LABEL_FAKE,
    LABEL_REAL,
    OPTIONAL_MANIFEST_COLUMNS,
    REQUIRED_MANIFEST_COLUMNS,
    VALID_BINARY_LABELS,
    ManifestRow,
    load_manifest_csv,
    save_manifest_csv,
    validate_manifest_dataframe,
)
from prooflens.data.scanner import (
    DEFAULT_LABEL_MAPPING,
    DEFAULT_SPLIT_MAPPING,
    ParsedRelativePath,
    parse_relative_image_path,
    scan_dataset_directory,
)
from prooflens.data.validation import (
    DEFAULT_SUPPORTED_EXTENSIONS,
    ImageIntegrityResult,
    compute_file_sha256,
    summarize_manifest_audit,
    verify_image_file,
)

__all__ = [
    "ALL_MANIFEST_COLUMNS",
    "DEFAULT_LABEL_MAPPING",
    "DEFAULT_SPLIT_MAPPING",
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "ImageIntegrityResult",
    "LABEL_FAKE",
    "LABEL_REAL",
    "ManifestRow",
    "OPTIONAL_MANIFEST_COLUMNS",
    "ParsedRelativePath",
    "ProofLensDataset",
    "REQUIRED_MANIFEST_COLUMNS",
    "VALID_BINARY_LABELS",
    "build_image_transforms",
    "compute_file_sha256",
    "create_dataloader",
    "load_manifest_csv",
    "parse_relative_image_path",
    "save_manifest_csv",
    "scan_dataset_directory",
    "summarize_manifest_audit",
    "validate_manifest_dataframe",
    "verify_image_file",
]
