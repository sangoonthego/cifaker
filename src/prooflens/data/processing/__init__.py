"""Image bytes validation, immutable TAR shard writing, and incremental materialization."""

from __future__ import annotations

from prooflens.data.processing.materializer import (
    MaterializationResult,
    PlannedShardSpec,
    materialize_release,
    plan_deterministic_shards,
)
from prooflens.data.processing.shard_writer import (
    ImmutableTarShardWriter,
    ShardImmutabilityError,
    ShardMemberEntry,
    ShardReceipt,
    compute_file_sha256,
    verify_tar_shard_file,
)
from prooflens.data.processing.validator import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ByteValidationResult,
    compute_dhash64,
    validate_image_bytes,
)

__all__ = [
    "ByteValidationResult",
    "ImmutableTarShardWriter",
    "MaterializationResult",
    "PlannedShardSpec",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "ShardImmutabilityError",
    "ShardMemberEntry",
    "ShardReceipt",
    "compute_dhash64",
    "compute_file_sha256",
    "materialize_release",
    "plan_deterministic_shards",
    "validate_image_bytes",
    "verify_tar_shard_file",
]
