"""Storage backend abstractions and factory for ProofLens."""

from __future__ import annotations

from prooflens.data.config import StorageConfig
from prooflens.data.storage.base import (
    ImmutabilityError,
    StorageAuthenticationError,
    StorageBackend,
    StorageCapacityInfo,
    StorageError,
    StorageObjectMetadata,
    compute_local_hashes,
    normalize_storage_key,
)
from prooflens.data.storage.google_drive import GoogleDriveStorage
from prooflens.data.storage.local import LocalFilesystemStorage


def create_storage_backend(config: StorageConfig) -> StorageBackend:
    """Instantiate a configured storage backend (`local` or `google_drive`)."""
    if config.backend == "local":
        assert config.root_path is not None
        return LocalFilesystemStorage(
            root_dir=config.root_path,
            verify_after_upload=config.verify_after_upload,
        )
    if config.backend == "google_drive":
        return GoogleDriveStorage(
            mode=config.gdrive_mode,
            folder_id=config.gdrive_folder_id,
            folder_name=config.gdrive_folder_name,
            mount_root=config.gdrive_mount_root,
            max_retries=config.max_retries,
            retry_backoff_seconds=config.retry_backoff_seconds,
            verify_after_upload=config.verify_after_upload,
        )
    raise ValueError(f"Unsupported storage backend: {config.backend!r}")


__all__ = [
    "GoogleDriveStorage",
    "ImmutabilityError",
    "LocalFilesystemStorage",
    "StorageAuthenticationError",
    "StorageBackend",
    "StorageCapacityInfo",
    "StorageError",
    "StorageObjectMetadata",
    "compute_local_hashes",
    "create_storage_backend",
    "normalize_storage_key",
]
