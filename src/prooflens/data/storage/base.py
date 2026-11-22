"""Minimal, replaceable storage backend abstraction for ProofLens dataset releases.

Designed so Google Drive is the initial remote storage provider and can later be
replaced by Google Cloud Storage (GCS) or S3 without changing inventory, sampling,
materialization, verification, or consumer code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class StorageError(RuntimeError):
    """Base error raised by ProofLens storage backends."""


class ImmutabilityError(StorageError):
    """Raised when attempting to overwrite or delete an immutable committed shard or published release."""


class StorageAuthenticationError(StorageError):
    """Raised when remote storage credentials are missing or invalid."""


def normalize_storage_key(key: str) -> str:
    """Normalize a relative object key and block path traversal or absolute paths."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Storage key must be a non-empty string.")
    norm = key.strip().replace("\\", "/")
    if norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        raise ValueError(f"Storage key must be relative, got: {key!r}")
    if "\x00" in norm:
        raise ValueError(f"NUL byte rejected in storage key: {key!r}")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"Unsafe storage key traversal rejected: {key!r}")
    return "/".join(parts)


def compute_local_hashes(path: Path, chunk_size: int = 1024 * 1024) -> tuple[int, str, str]:
    """Compute `(byte_size, sha256_hex, md5_hex)` of a local file in a single streaming pass."""
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            sha.update(chunk)
            md5.update(chunk)
            total += len(chunk)
    return total, sha.hexdigest(), md5.hexdigest()


@dataclass(frozen=True)
class StorageObjectMetadata:
    """Metadata describing a stored object."""

    key: str
    size_bytes: int
    sha256: str | None = None
    md5: str | None = None
    modified_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StorageBackend(ABC):
    """Abstract storage interface supporting local disk, Google Drive, and future cloud stores."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable identifier of the storage backend."""

    @abstractmethod
    def upload(
        self,
        local_path: Path | str,
        remote_key: str,
        expected_sha256: str | None = None,
        overwrite: bool = False,
    ) -> StorageObjectMetadata:
        """Upload a local file to `remote_key`, verifying integrity and enforcing immutability."""

    @abstractmethod
    def download(
        self,
        remote_key: str,
        local_path: Path | str,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download `remote_key` to `local_path`, optionally verifying SHA-256."""

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Return True if an object exists at `remote_key`."""

    @abstractmethod
    def get_metadata(self, remote_key: str) -> StorageObjectMetadata:
        """Fetch size and checksum metadata for an existing object."""

    @abstractmethod
    def verify_integrity(
        self,
        remote_key: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> bool:
        """Verify that `remote_key` exists and matches expected size/checksum."""

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[str]:
        """List relative object keys under `prefix` in deterministic sorted order."""

    @abstractmethod
    def delete_uncommitted(self, remote_key: str) -> None:
        """Delete an uncommitted/staging object (strictly forbidden on published releases)."""

    def is_committed_release(self, release_id: str) -> bool:
        """Return True if `releases/{release_id}/release.json` exists (published release marker)."""
        return self.exists(f"releases/{release_id}/release.json")

    def read_bytes(self, remote_key: str) -> bytes:
        """Convenience helper returning the small object payload as bytes."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "obj.bin"
            self.download(remote_key, dst)
            return dst.read_bytes()

    def write_bytes(
        self,
        data: bytes,
        remote_key: str,
        overwrite: bool = False,
    ) -> StorageObjectMetadata:
        """Convenience helper uploading an in-memory small payload (JSON/state) with verification."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "obj.bin"
            src.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            return self.upload(src, remote_key, expected_sha256=sha, overwrite=overwrite)
