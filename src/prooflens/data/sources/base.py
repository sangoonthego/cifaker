"""Source adapter contracts separating upstream datasets from downstream processing.

Supports directory trees, archive shards (TAR/ZIP), and Hugging Face datasets
with explicit capability declarations so callers never assume random access when
only sequential or shard-aware streaming is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any


class SourceAccessError(RuntimeError):
    """Raised when an upstream source record or image cannot be read."""


@dataclass(frozen=True)
class SourceCapabilities:
    """Explicit operational capabilities exposed by a SourceAdapter."""

    metadata_only_enumeration: bool
    sequential_streaming: bool
    random_access: bool
    shard_level_access: bool
    resume_support: bool
    estimated_transfer_bytes: int | None = None
    access_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceRecord:
    """Upstream image metadata discovered during source enumeration."""

    source_id: str
    source_revision: str
    source_path: str
    source_split: str
    label: int
    generator: str
    source_generator_group: str
    source_shard: str | None = None
    imagenet_class: str | None = None
    file_size_bytes: int | None = None
    file_extension: str = ""
    original_format: str | None = None


@dataclass(frozen=True)
class SourceImagePayload:
    """Original raw bytes and provenance returned when fetching a source image."""

    source_path: str
    raw_bytes: bytes
    file_extension: str
    source_shard: str | None = None


class SourceAdapter(ABC):
    """Abstract interface for accessing an upstream GenImage or compatible dataset source."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Stable identifier of the upstream dataset source."""

    @property
    @abstractmethod
    def source_revision(self) -> str:
        """Immutable version, snapshot tag, or commit SHA of the source."""

    @abstractmethod
    def capabilities(self) -> SourceCapabilities:
        """Inspect supported access patterns and transfer constraints."""

    @abstractmethod
    def iter_records(self) -> Iterator[SourceRecord]:
        """Stream metadata records deterministically without loading all images into RAM."""

    @abstractmethod
    def fetch_image_bytes(self, record: SourceRecord) -> SourceImagePayload:
        """Fetch raw original image bytes for a single record (requires random or cached access)."""

    def iter_selected_images(
        self,
        records: Iterable[SourceRecord],
    ) -> Iterator[tuple[SourceRecord, SourceImagePayload | None, str | None]]:
        """Yield `(record, payload, error_message)` for a batch of selected records.

        Subclasses without random access (such as sequential TAR archives or HF streaming)
        override this method to perform shard-aware or single-pass streaming reads instead
        of downloading an entire archive per image.
        """
        for rec in records:
            try:
                payload = self.fetch_image_bytes(rec)
                yield rec, payload, None
            except Exception as exc:
                yield rec, None, f"{type(exc).__name__}: {exc}"
