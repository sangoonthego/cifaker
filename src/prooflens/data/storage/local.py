"""Local filesystem storage backend for development, CI, and offline integration tests."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from prooflens.data.storage.base import (
    ImmutabilityError,
    StorageBackend,
    StorageCapacityInfo,
    StorageError,
    StorageObjectMetadata,
    compute_local_hashes,
    normalize_storage_key,
)


class LocalFilesystemStorage(StorageBackend):
    """POSIX-staged local filesystem storage implementing immutable release semantics."""

    def __init__(self, root_dir: Path | str, verify_after_upload: bool = True) -> None:
        self._root = Path(root_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._verify_after_upload = verify_after_upload

    @property
    def backend_name(self) -> str:
        return "local_filesystem"

    @property
    def root_dir(self) -> Path:
        return self._root

    def _resolve_key(self, remote_key: str) -> Path:
        norm = normalize_storage_key(remote_key)
        target = (self._root / norm).resolve()
        if not str(target).startswith(str(self._root)):
            raise StorageError(f"Storage key escapes root directory: {remote_key!r}")
        return target

    def _extract_release_id(self, norm_key: str) -> str | None:
        parts = norm_key.split("/")
        if len(parts) >= 3 and parts[0] == "releases":
            return parts[1]
        return None

    def _guard_immutability(self, norm_key: str, overwrite: bool) -> None:
        rel_id = self._extract_release_id(norm_key)
        if rel_id is not None and self.is_committed_release(rel_id):
            raise ImmutabilityError(
                f"Release '{rel_id}' is already PUBLISHED and immutable; cannot mutate '{norm_key}'."
            )
        target = self._resolve_key(norm_key)
        if not overwrite and target.exists():
            raise ImmutabilityError(
                f"Refusing to overwrite existing immutable object at '{norm_key}'."
            )

    def upload(
        self,
        local_path: Path | str,
        remote_key: str,
        expected_sha256: str | None = None,
        overwrite: bool = False,
    ) -> StorageObjectMetadata:
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(f"Upload source file does not exist: {src}")

        norm_key = normalize_storage_key(remote_key)
        self._guard_immutability(norm_key, overwrite=overwrite)

        src_size, src_sha, src_md5 = compute_local_hashes(src)
        if expected_sha256 is not None and src_sha != expected_sha256:
            raise StorageError(
                f"Source file SHA-256 mismatch before upload: expected {expected_sha256}, got {src_sha}"
            )

        dst = self._resolve_key(norm_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        staging = dst.with_name(f".{dst.name}.upload.tmp")
        try:
            shutil.copy2(src, staging)
            if self._verify_after_upload:
                stg_size, stg_sha, _ = compute_local_hashes(staging)
                if stg_size != src_size or stg_sha != src_sha:
                    raise StorageError(
                        f"Post-copy integrity check failed for '{norm_key}': "
                        f"expected ({src_size}, {src_sha}), got ({stg_size}, {stg_sha})"
                    )
            staging.replace(dst)
        finally:
            staging.unlink(missing_ok=True)

        mtime = datetime.fromtimestamp(dst.stat().st_mtime, tz=UTC).isoformat()
        return StorageObjectMetadata(
            key=norm_key,
            size_bytes=src_size,
            sha256=src_sha,
            md5=src_md5,
            modified_at_utc=mtime,
        )

    def download(
        self,
        remote_key: str,
        local_path: Path | str,
        expected_sha256: str | None = None,
    ) -> Path:
        norm_key = normalize_storage_key(remote_key)
        src = self._resolve_key(norm_key)
        if not src.is_file():
            raise FileNotFoundError(f"Storage object not found: '{norm_key}'")

        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        staging = dst.with_name(f".{dst.name}.download.tmp")
        try:
            shutil.copy2(src, staging)
            if expected_sha256 is not None:
                _, actual_sha, _ = compute_local_hashes(staging)
                if actual_sha != expected_sha256:
                    raise StorageError(
                        f"Downloaded object '{norm_key}' SHA-256 mismatch: "
                        f"expected {expected_sha256}, got {actual_sha}"
                    )
            staging.replace(dst)
        finally:
            staging.unlink(missing_ok=True)

        return dst

    def exists(self, remote_key: str) -> bool:
        norm_key = normalize_storage_key(remote_key)
        return self._resolve_key(norm_key).is_file()

    def get_metadata(self, remote_key: str) -> StorageObjectMetadata:
        norm_key = normalize_storage_key(remote_key)
        target = self._resolve_key(norm_key)
        if not target.is_file():
            raise FileNotFoundError(f"Storage object not found: '{norm_key}'")
        size, sha, md5 = compute_local_hashes(target)
        mtime = datetime.fromtimestamp(target.stat().st_mtime, tz=UTC).isoformat()
        return StorageObjectMetadata(
            key=norm_key,
            size_bytes=size,
            sha256=sha,
            md5=md5,
            modified_at_utc=mtime,
        )

    def verify_integrity(
        self,
        remote_key: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> bool:
        if not self.exists(remote_key):
            return False
        meta = self.get_metadata(remote_key)
        if expected_size is not None and meta.size_bytes != expected_size:
            return False
        if expected_sha256 is not None and meta.sha256 != expected_sha256:
            return False
        return True

    def list_objects(self, prefix: str = "") -> list[str]:
        base = self._resolve_key(prefix) if prefix.strip() else self._root
        if not base.exists():
            return []
        if base.is_file():
            return [base.relative_to(self._root).as_posix()]

        keys: list[str] = []
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".upload.tmp"):
                keys.append(path.relative_to(self._root).as_posix())
        return keys

    def delete_uncommitted(self, remote_key: str) -> None:
        norm_key = normalize_storage_key(remote_key)
        rel_id = self._extract_release_id(norm_key)
        if rel_id is not None and self.is_committed_release(rel_id):
            raise ImmutabilityError(
                f"Cannot delete '{norm_key}': release '{rel_id}' is already PUBLISHED."
            )
        target = self._resolve_key(norm_key)
        target.unlink(missing_ok=True)

    def get_capacity_info(self) -> StorageCapacityInfo:
        usage = shutil.disk_usage(self._root)
        return StorageCapacityInfo(
            backend_name=self.backend_name,
            status="KNOWN",
            total_bytes=int(usage.total),
            used_bytes=int(usage.used),
            free_bytes=int(usage.free),
            notes=f"Local filesystem capacity queried at '{self._root}'.",
        )
