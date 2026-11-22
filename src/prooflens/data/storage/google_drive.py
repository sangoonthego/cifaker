"""Google Drive storage backend for ProofLens (API v3 mode + Mounted Drive mode).

Target shared folder convention:
- Folder name: `cifaker` (configurable)
- Folder ID: configurable via YAML (e.g., `1IipfeWX_OVJS2KwaFL5wFlO-edxLoEGD`)

Important operational guarantees & honest limitations:
1. Clearly distinguishes `api` mode (Google Drive API v3 via `google-api-python-client`
   + interactive Colab OAuth / ADC without storing secrets in code) from `mounted` mode
   (`/content/drive/MyDrive/...` via Colab Drive FUSE).
2. Does NOT assume mounted Google Drive is a transactional POSIX filesystem:
   `mounted` mode stages to a temp file, calls `os.fsync`, promotes, and performs an
   explicit read-back SHA-256 verification with retries.
3. `api` mode verifies every uploaded object against Google Drive's remote `size` and
   `md5Checksum` metadata, retrying transient failures with exponential backoff.
4. Enforces single-writer publication semantics and immutability of published releases,
   while explicitly acknowledging that Google Drive lacks atomic cross-file transactions.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from prooflens.data.storage.base import (
    ImmutabilityError,
    StorageAuthenticationError,
    StorageBackend,
    StorageError,
    StorageObjectMetadata,
    compute_local_hashes,
    normalize_storage_key,
)

DRIVE_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/drive",)


class GoogleDriveStorage(StorageBackend):
    """Google Drive storage backend supporting explicit `api` and `mounted` modes."""

    def __init__(
        self,
        mode: str = "api",
        folder_id: str | None = None,
        folder_name: str = "cifaker",
        mount_root: Path | str | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        verify_after_upload: bool = True,
        drive_service: Any | None = None,
    ) -> None:
        if mode not in {"api", "mounted"}:
            raise ValueError(f"GoogleDriveStorage mode must be 'api' or 'mounted', got {mode!r}")
        if mode == "api" and not folder_id:
            raise ValueError("GoogleDriveStorage(mode='api') requires a configured 'folder_id'.")
        if mode == "mounted" and not mount_root:
            raise ValueError(
                "GoogleDriveStorage(mode='mounted') requires a configured 'mount_root'."
            )

        self._mode = mode
        self._folder_id = folder_id
        self._folder_name = folder_name
        self._mount_root = Path(mount_root).resolve() if mount_root is not None else None
        self._max_retries = max(1, int(max_retries))
        self._backoff = max(0.0, float(retry_backoff_seconds))
        self._verify_after_upload = verify_after_upload
        self._service = drive_service
        self._folder_cache: dict[tuple[str, str], str] = {}

    @property
    def backend_name(self) -> str:
        return f"google_drive_{self._mode}"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def folder_id(self) -> str | None:
        return self._folder_id

    def _extract_release_id(self, norm_key: str) -> str | None:
        parts = norm_key.split("/")
        if len(parts) >= 3 and parts[0] == "releases":
            return parts[1]
        return None

    def _retry(self, op_name: str, fn: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return fn()
            except (ImmutabilityError, StorageAuthenticationError, FileNotFoundError):
                raise
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self._max_retries and self._backoff > 0:
                    time.sleep(self._backoff * (2**attempt))
        raise StorageError(
            f"GoogleDriveStorage.{op_name} failed after {self._max_retries} attempt(s): {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Google Drive API v3 Helpers (mode == "api")
    # ------------------------------------------------------------------

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service

        # Interactive Colab authentication if running inside Google Colab
        try:
            from google.colab import auth as colab_auth  # type: ignore[import-not-found]

            colab_auth.authenticate_user()
        except ImportError:
            pass

        try:
            creds, _ = google.auth.default(scopes=list(DRIVE_SCOPES))
        except Exception as exc:
            raise StorageAuthenticationError(
                "Google Drive credentials are unavailable. On Google Colab, run interactive "
                "authentication (`google.colab.auth.authenticate_user()`). Locally, configure "
                "Application Default Credentials via `gcloud auth application-default login` or "
                "`GOOGLE_APPLICATION_CREDENTIALS` (never commit credential files)."
            ) from exc

        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def _find_child(
        self,
        parent_id: str,
        name: str,
        mime_type: str | None = None,
    ) -> dict[str, Any] | None:
        service = self._get_service()
        escaped = name.replace("'", "\\'")
        query = f"'{parent_id}' in parents and name = '{escaped}' and trashed = false"
        if mime_type is not None:
            query += f" and mimeType = '{mime_type}'"
        resp = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name, size, md5Checksum, modifiedTime, mimeType)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageSize=10,
            )
            .execute()
        )
        files = resp.get("files", [])
        if not files:
            return None
        return files[0]

    def _ensure_folder_path(self, parts: list[str], create: bool = True) -> str | None:
        assert self._folder_id is not None
        parent_id = self._folder_id
        for part in parts:
            cache_key = (parent_id, part)
            if cache_key in self._folder_cache:
                parent_id = self._folder_cache[cache_key]
                continue
            existing = self._find_child(
                parent_id, part, mime_type="application/vnd.google-apps.folder"
            )
            if existing is not None:
                folder_id = str(existing["id"])
                self._folder_cache[cache_key] = folder_id
                parent_id = folder_id
                continue
            if not create:
                return None
            service = self._get_service()
            created = (
                service.files()
                .create(
                    body={
                        "name": part,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [parent_id],
                    },
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            folder_id = str(created["id"])
            self._folder_cache[cache_key] = folder_id
            parent_id = folder_id
        return parent_id

    def _lookup_api_file(self, norm_key: str) -> dict[str, Any] | None:
        parts = norm_key.split("/")
        parent_id = self._ensure_folder_path(parts[:-1], create=False)
        if parent_id is None:
            return None
        return self._find_child(parent_id, parts[-1])

    # ------------------------------------------------------------------
    # Mounted Drive Helpers (mode == "mounted")
    # ------------------------------------------------------------------

    def _resolve_mounted(self, norm_key: str) -> Path:
        assert self._mount_root is not None
        target = (self._mount_root / norm_key).resolve()
        if not str(target).startswith(str(self._mount_root)):
            raise StorageError(f"Key escapes mounted Drive root: {norm_key!r}")
        return target

    def _copy_and_fsync(self, src: Path, dst: Path, chunk_size: int = 1024 * 1024) -> None:
        with src.open("rb") as in_f, dst.open("wb") as out_f:
            shutil.copyfileobj(in_f, out_f, length=chunk_size)
            out_f.flush()
            os.fsync(out_f.fileno())

    # ------------------------------------------------------------------
    # Unified StorageBackend Contract
    # ------------------------------------------------------------------

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
        rel_id = self._extract_release_id(norm_key)
        if rel_id is not None and self.is_committed_release(rel_id):
            raise ImmutabilityError(
                f"Release '{rel_id}' is already PUBLISHED on Google Drive and immutable."
            )

        src_size, src_sha, src_md5 = compute_local_hashes(src)
        if expected_sha256 is not None and src_sha != expected_sha256:
            raise StorageError(
                f"Local SHA-256 mismatch before Drive upload: expected {expected_sha256}, got {src_sha}"
            )

        if self._mode == "mounted":
            return self._upload_mounted(src, norm_key, src_size, src_sha, src_md5, overwrite)
        return self._upload_api(src, norm_key, src_size, src_sha, src_md5, overwrite)

    def _upload_mounted(
        self,
        src: Path,
        norm_key: str,
        src_size: int,
        src_sha: str,
        src_md5: str,
        overwrite: bool,
    ) -> StorageObjectMetadata:
        dst = self._resolve_mounted(norm_key)
        if not overwrite and dst.exists():
            raise ImmutabilityError(
                f"Refusing to overwrite existing object on mounted Drive: '{norm_key}'."
            )

        def _do_copy() -> StorageObjectMetadata:
            dst.parent.mkdir(parents=True, exist_ok=True)
            staging = dst.with_name(f".{dst.name}.gdrive_stage.tmp")
            try:
                self._copy_and_fsync(src, staging)
                if dst.exists() and overwrite:
                    dst.unlink()
                staging.replace(dst)
                # Explicit read-back verification (Drive FUSE is non-transactional)
                if self._verify_after_upload:
                    rb_size, rb_sha, _ = compute_local_hashes(dst)
                    if rb_size != src_size or rb_sha != src_sha:
                        raise StorageError(
                            f"Mounted Drive read-back verification failed for '{norm_key}': "
                            f"expected ({src_size}, {src_sha}), got ({rb_size}, {rb_sha})"
                        )
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

        return self._retry(f"upload_mounted({norm_key})", _do_copy)

    def _upload_api(
        self,
        src: Path,
        norm_key: str,
        src_size: int,
        src_sha: str,
        src_md5: str,
        overwrite: bool,
    ) -> StorageObjectMetadata:
        def _do_api_upload() -> StorageObjectMetadata:
            service = self._get_service()
            existing = self._lookup_api_file(norm_key)
            if existing is not None and not overwrite:
                raise ImmutabilityError(
                    f"Refusing to overwrite existing object on Google Drive API: '{norm_key}'."
                )

            parts = norm_key.split("/")
            parent_id = self._ensure_folder_path(parts[:-1], create=True)
            assert parent_id is not None

            media = MediaFileUpload(str(src), resumable=True)

            if existing is not None and overwrite:
                uploaded = (
                    service.files()
                    .update(
                        fileId=existing["id"],
                        media_body=media,
                        fields="id, name, size, md5Checksum, modifiedTime",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
            else:
                uploaded = (
                    service.files()
                    .create(
                        body={"name": parts[-1], "parents": [parent_id]},
                        media_body=media,
                        fields="id, name, size, md5Checksum, modifiedTime",
                        supportsAllDrives=True,
                    )
                    .execute()
                )

            remote_size = int(uploaded.get("size", -1))
            remote_md5 = str(uploaded.get("md5Checksum", "")).lower()
            if self._verify_after_upload:
                if remote_size != src_size:
                    raise StorageError(
                        f"Google Drive upload size mismatch for '{norm_key}': "
                        f"expected {src_size}, got {remote_size}"
                    )
                if remote_md5 and remote_md5 != src_md5:
                    raise StorageError(
                        f"Google Drive upload MD5 mismatch for '{norm_key}': "
                        f"expected {src_md5}, got {remote_md5}"
                    )

            return StorageObjectMetadata(
                key=norm_key,
                size_bytes=remote_size if remote_size >= 0 else src_size,
                sha256=src_sha,
                md5=remote_md5 or src_md5,
                modified_at_utc=uploaded.get("modifiedTime"),
            )

        return self._retry(f"upload_api({norm_key})", _do_api_upload)

    def download(
        self,
        remote_key: str,
        local_path: Path | str,
        expected_sha256: str | None = None,
    ) -> Path:
        norm_key = normalize_storage_key(remote_key)
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if self._mode == "mounted":
            src = self._resolve_mounted(norm_key)
            if not src.is_file():
                raise FileNotFoundError(f"Object not found on mounted Drive: '{norm_key}'")

            def _do_mount_dl() -> Path:
                staging = dst.with_name(f".{dst.name}.dl.tmp")
                try:
                    shutil.copyfile(src, staging)
                    if expected_sha256 is not None:
                        _, actual_sha, _ = compute_local_hashes(staging)
                        if actual_sha != expected_sha256:
                            raise StorageError(
                                f"SHA-256 mismatch downloading '{norm_key}' from mounted Drive: "
                                f"expected {expected_sha256}, got {actual_sha}"
                            )
                    staging.replace(dst)
                finally:
                    staging.unlink(missing_ok=True)
                return dst

            return self._retry(f"download_mounted({norm_key})", _do_mount_dl)

        def _do_api_dl() -> Path:
            service = self._get_service()
            meta = self._lookup_api_file(norm_key)
            if meta is None:
                raise FileNotFoundError(f"Object not found on Google Drive API: '{norm_key}'")

            staging = dst.with_name(f".{dst.name}.dl.tmp")
            try:
                request = service.files().get_media(fileId=meta["id"], supportsAllDrives=True)
                if hasattr(request, "http"):
                    with staging.open("wb") as handle:
                        downloader = MediaIoBaseDownload(handle, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                else:
                    data = request.execute()
                    staging.write_bytes(data)

                if expected_sha256 is not None:
                    _, actual_sha, _ = compute_local_hashes(staging)
                    if actual_sha != expected_sha256:
                        raise StorageError(
                            f"SHA-256 mismatch downloading '{norm_key}' from Drive API: "
                            f"expected {expected_sha256}, got {actual_sha}"
                        )
                staging.replace(dst)
            finally:
                staging.unlink(missing_ok=True)
            return dst

        return self._retry(f"download_api({norm_key})", _do_api_dl)

    def exists(self, remote_key: str) -> bool:
        norm_key = normalize_storage_key(remote_key)
        if self._mode == "mounted":
            return self._resolve_mounted(norm_key).is_file()
        return self._retry(
            f"exists({norm_key})", lambda: self._lookup_api_file(norm_key) is not None
        )

    def get_metadata(self, remote_key: str) -> StorageObjectMetadata:
        norm_key = normalize_storage_key(remote_key)
        if self._mode == "mounted":
            target = self._resolve_mounted(norm_key)
            if not target.is_file():
                raise FileNotFoundError(f"Object not found on mounted Drive: '{norm_key}'")
            size, sha, md5 = compute_local_hashes(target)
            mtime = datetime.fromtimestamp(target.stat().st_mtime, tz=UTC).isoformat()
            return StorageObjectMetadata(
                key=norm_key, size_bytes=size, sha256=sha, md5=md5, modified_at_utc=mtime
            )

        meta = self._retry(f"get_metadata({norm_key})", lambda: self._lookup_api_file(norm_key))
        if meta is None:
            raise FileNotFoundError(f"Object not found on Google Drive API: '{norm_key}'")
        return StorageObjectMetadata(
            key=norm_key,
            size_bytes=int(meta.get("size", 0)),
            sha256=None,
            md5=str(meta.get("md5Checksum", "")).lower() or None,
            modified_at_utc=meta.get("modifiedTime"),
        )

    def verify_integrity(
        self,
        remote_key: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> bool:
        if not self.exists(remote_key):
            return False
        if self._mode == "mounted":
            meta = self.get_metadata(remote_key)
            if expected_size is not None and meta.size_bytes != expected_size:
                return False
            if expected_sha256 is not None and meta.sha256 != expected_sha256:
                return False
            return True

        meta = self.get_metadata(remote_key)
        if expected_size is not None and meta.size_bytes != expected_size:
            return False
        # Drive API v3 stores MD5 natively rather than SHA-256; if exact SHA-256 is required,
        # stream-verify via download into a bounded temporary hash check.
        if expected_sha256 is not None:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                try:
                    self.download(
                        remote_key, Path(tmp) / "verify.bin", expected_sha256=expected_sha256
                    )
                except StorageError:
                    return False
        return True

    def list_objects(self, prefix: str = "") -> list[str]:
        if self._mode == "mounted":
            assert self._mount_root is not None
            base = (
                self._resolve_mounted(normalize_storage_key(prefix))
                if prefix.strip()
                else self._mount_root
            )
            if not base.exists():
                return []
            if base.is_file():
                return [base.relative_to(self._mount_root).as_posix()]
            return [
                p.relative_to(self._mount_root).as_posix()
                for p in sorted(base.rglob("*"))
                if p.is_file() and not p.name.endswith(".gdrive_stage.tmp")
            ]

        norm_prefix = normalize_storage_key(prefix) if prefix.strip() else ""
        parts = norm_prefix.split("/") if norm_prefix else []
        start_id = self._ensure_folder_path(parts, create=False)
        if start_id is None:
            return []

        service = self._get_service()
        results: list[str] = []

        def _walk(folder_id: str, rel_prefix: str) -> None:
            page_token = None
            while True:
                resp = (
                    service.files()
                    .list(
                        q=f"'{folder_id}' in parents and trashed = false",
                        spaces="drive",
                        fields="nextPageToken, files(id, name, mimeType)",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                        pageSize=500,
                        pageToken=page_token,
                    )
                    .execute()
                )
                for item in resp.get("files", []):
                    child_key = f"{rel_prefix}/{item['name']}" if rel_prefix else item["name"]
                    if item.get("mimeType") == "application/vnd.google-apps.folder":
                        _walk(item["id"], child_key)
                    else:
                        results.append(child_key)
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

        _walk(start_id, norm_prefix)
        results.sort()
        return results

    def delete_uncommitted(self, remote_key: str) -> None:
        norm_key = normalize_storage_key(remote_key)
        rel_id = self._extract_release_id(norm_key)
        if rel_id is not None and self.is_committed_release(rel_id):
            raise ImmutabilityError(
                f"Cannot delete '{norm_key}': release '{rel_id}' is already PUBLISHED on Drive."
            )
        if self._mode == "mounted":
            self._resolve_mounted(norm_key).unlink(missing_ok=True)
            return

        meta = self._lookup_api_file(norm_key)
        if meta is not None:
            service = self._get_service()
            self._retry(
                f"delete({norm_key})",
                lambda: service.files().delete(fileId=meta["id"], supportsAllDrives=True).execute(),
            )
