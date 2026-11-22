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

__all__ = [
    "ArchiveSourceAdapter",
    "DirectoryGenImageAdapter",
    "HuggingFaceGenImageAdapter",
    "SourceAccessError",
    "SourceAdapter",
    "SourceCapabilities",
    "SourceImagePayload",
    "SourceRecord",
    "create_source_adapter",
]
