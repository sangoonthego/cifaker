"""Dataset, manifest, scanner, validation, and scalable GenImage pipeline package for ProofLens."""

from __future__ import annotations

from prooflens.data.config import (
    ConfigValidationError,
    ProtocolConfig,
    ReleaseConfig,
    SourceConfig,
    SplitRuleConfig,
    StorageConfig,
    compute_config_hash,
    load_protocol_config,
    load_release_config,
    load_source_config,
    load_storage_config,
)
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
from prooflens.data.releases.manager import (
    BoundedShardCache,
    FullLocalShardCache,
    PreparedReleaseSplit,
    ReleaseConsumer,
    ReleaseManager,
    UnpublishedReleaseError,
    load_release,
)
from prooflens.data.scanner import (
    DEFAULT_LABEL_MAPPING,
    DEFAULT_SPLIT_MAPPING,
    ParsedRelativePath,
    parse_relative_image_path,
    scan_dataset_directory,
)
from prooflens.data.shard_dataset import (
    CorruptedShardMemberError,
    ShardIterableDataset,
    ShardMapDataset,
    create_shard_dataloader,
)
from prooflens.data.sources import create_source_adapter
from prooflens.data.storage import (
    GoogleDriveStorage,
    ImmutabilityError,
    LocalFilesystemStorage,
    create_storage_backend,
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
    "BoundedShardCache",
    "ConfigValidationError",
    "CorruptedShardMemberError",
    "DEFAULT_LABEL_MAPPING",
    "DEFAULT_SPLIT_MAPPING",
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "FullLocalShardCache",
    "GoogleDriveStorage",
    "ImageIntegrityResult",
    "ImmutabilityError",
    "LABEL_FAKE",
    "LABEL_REAL",
    "LocalFilesystemStorage",
    "ManifestRow",
    "OPTIONAL_MANIFEST_COLUMNS",
    "ParsedRelativePath",
    "PreparedReleaseSplit",
    "ProofLensDataset",
    "ProtocolConfig",
    "REQUIRED_MANIFEST_COLUMNS",
    "ReleaseConfig",
    "ReleaseConsumer",
    "ReleaseManager",
    "ShardIterableDataset",
    "ShardMapDataset",
    "SourceConfig",
    "SplitRuleConfig",
    "StorageConfig",
    "UnpublishedReleaseError",
    "VALID_BINARY_LABELS",
    "build_image_transforms",
    "compute_config_hash",
    "compute_file_sha256",
    "create_dataloader",
    "create_shard_dataloader",
    "create_source_adapter",
    "create_storage_backend",
    "load_manifest_csv",
    "load_protocol_config",
    "load_release",
    "load_release_config",
    "load_source_config",
    "load_storage_config",
    "parse_relative_image_path",
    "save_manifest_csv",
    "scan_dataset_directory",
    "summarize_manifest_audit",
    "validate_manifest_dataframe",
    "verify_image_file",
]
