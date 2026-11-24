"""Upstream dataset source adapters for ProofLens."""

from __future__ import annotations

from prooflens.data.sources.archive import ArchiveSourceAdapter
from prooflens.data.sources.base import (
    SourceAccessError,
    SourceAdapter,
    SourceCapabilities,
    SourceImagePayload,
    SourceRecord,
)
from prooflens.data.sources.genimage import (
    DirectoryGenImageAdapter,
    HuggingFaceGenImageAdapter,
    create_source_adapter,
)
from prooflens.data.sources.genimage_arrow import (
    DEFAULT_ARROW_GENERATOR_MAPPING,
    ArrowLocatorError,
    ArrowSchemaError,
    GenImageArrowAdapter,
    ParsedArrowLocator,
    ParsedArrowShard,
    build_arrow_source_path,
    parse_arrow_shard_path,
    parse_arrow_source_path,
)

__all__ = [
    "ArchiveSourceAdapter",
    "ArrowLocatorError",
    "ArrowSchemaError",
    "DEFAULT_ARROW_GENERATOR_MAPPING",
    "DirectoryGenImageAdapter",
    "GenImageArrowAdapter",
    "HuggingFaceGenImageAdapter",
    "ParsedArrowLocator",
    "ParsedArrowShard",
    "SourceAccessError",
    "SourceAdapter",
    "SourceCapabilities",
    "SourceImagePayload",
    "SourceRecord",
    "build_arrow_source_path",
    "create_source_adapter",
    "parse_arrow_shard_path",
    "parse_arrow_source_path",
]
