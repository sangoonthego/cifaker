"""Immutable uncompressed TAR shard writer with staging, checksums, and member offsets.

Design properties:
- Writes original image bytes & original extensions verbatim (no resizing or re-encoding).
- Uses uncompressed POSIX TAR so downstream PyTorch readers can seek directly to
  `(offset_data, byte_size)` in O(1) without extracting thousands of individual files.
- Stages writes to `<shard_id>.tar.tmp`, computes exact SHA-256, verifies readability,
  and atomically promotes to `<shard_id>.tar` + `<shard_id>.receipt.json`.
- Refuses to overwrite an already-committed shard (`ShardImmutabilityError`).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ShardImmutabilityError(RuntimeError):
    """Raised when attempting to overwrite an already committed immutable TAR shard."""


@dataclass(frozen=True)
class ShardMemberEntry:
    """Inventory record for a single image stored inside a TAR shard."""

    image_id: str
    member_name: str
    offset_data: int
    byte_size: int
    sha256: str
    label: int
    generator: str
    source_generator_group: str
    source_split: str
    release_split: str
    width: int | None
    height: int | None
    image_format: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShardReceipt:
    """Verified processing receipt persisted alongside every committed TAR shard."""

    shard_id: str
    release_id: str
    origin_release_id: str
    release_split: str
    storage_key: str
    status: str
    image_count: int
    byte_size: int
    sha256: str
    created_at_utc: str
    members: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardReceipt:
        return cls(
            shard_id=str(data["shard_id"]),
            release_id=str(data["release_id"]),
            origin_release_id=str(data.get("origin_release_id", data["release_id"])),
            release_split=str(data["release_split"]),
            storage_key=str(data["storage_key"]),
            status=str(data["status"]),
            image_count=int(data["image_count"]),
            byte_size=int(data["byte_size"]),
            sha256=str(data["sha256"]),
            created_at_utc=str(data["created_at_utc"]),
            members=list(data.get("members", [])),
        )


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 checksum of a file on disk with bounded memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_tar_shard_file(
    tar_path: Path,
    expected_sha256: str | None = None,
    expected_byte_size: int | None = None,
    expected_image_count: int | None = None,
) -> tuple[bool, str]:
    """Verify a TAR shard file on disk (size, SHA-256, and readable TAR member headers)."""
    if not tar_path.is_file():
        return False, f"Shard file missing: {tar_path}"

    actual_size = tar_path.stat().st_size
    if expected_byte_size is not None and actual_size != expected_byte_size:
        return False, f"Shard byte_size mismatch: expected {expected_byte_size}, got {actual_size}"

    if expected_sha256 is not None:
        actual_sha = compute_file_sha256(tar_path)
        if actual_sha != expected_sha256:
            return False, f"Shard SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"

    try:
        member_count = 0
        with tarfile.open(tar_path, mode="r:") as tf:
            for member in tf:
                if member.isfile():
                    member_count += 1
        if expected_image_count is not None and member_count != expected_image_count:
            return (
                False,
                f"Shard member count mismatch: expected {expected_image_count}, got {member_count}",
            )
    except Exception as exc:
        return False, f"Corrupted TAR archive ({type(exc).__name__}): {exc}"

    return True, "OK"


class ImmutableTarShardWriter:
    """Staged writer for creating a single immutable TAR shard and its verified receipt."""

    def __init__(
        self,
        shard_dir: Path | str,
        shard_id: str,
        release_id: str,
        release_split: str,
        storage_key: str,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_id = shard_id
        self.release_id = release_id
        self.release_split = release_split
        self.storage_key = storage_key

        self.final_tar_path = self.shard_dir / f"{shard_id}.tar"
        self.receipt_path = self.shard_dir / f"{shard_id}.receipt.json"
        self.staging_tar_path = self.shard_dir / f"{shard_id}.tar.tmp"

        if self.final_tar_path.exists() and self.receipt_path.exists():
            raise ShardImmutabilityError(
                f"Refusing to overwrite committed shard '{shard_id}' at {self.final_tar_path}."
            )

        # Clean up any stale staging file from a previously interrupted run
        if self.staging_tar_path.exists():
            self.staging_tar_path.unlink()

        self._tar: tarfile.TarFile | None = tarfile.open(self.staging_tar_path, mode="w:")
        self._members: list[ShardMemberEntry] = []
        self._finalized = False

    def add_image(
        self,
        image_id: str,
        raw_bytes: bytes,
        file_extension: str,
        sha256: str,
        label: int,
        generator: str,
        source_generator_group: str,
        source_split: str,
        width: int | None = None,
        height: int | None = None,
        image_format: str | None = None,
    ) -> ShardMemberEntry:
        """Append one image's original raw bytes deterministically into the staging TAR shard."""
        if self._finalized or self._tar is None:
            raise RuntimeError("Cannot add images to a finalized or aborted shard writer.")

        ext = file_extension.lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}" if ext else ".bin"
        member_name = f"{image_id}{ext}"

        info = tarfile.TarInfo(name=member_name)
        info.size = len(raw_bytes)
        # Fixed deterministic header metadata so TAR bytes are reproducible across runs
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mode = 0o644

        self._tar.addfile(info, io.BytesIO(raw_bytes))

        entry = ShardMemberEntry(
            image_id=image_id,
            member_name=member_name,
            offset_data=int(info.offset_data),
            byte_size=len(raw_bytes),
            sha256=sha256,
            label=int(label),
            generator=generator,
            source_generator_group=source_generator_group,
            source_split=source_split,
            release_split=self.release_split,
            width=width,
            height=height,
            image_format=image_format,
        )
        self._members.append(entry)
        return entry

    def finalize(self) -> tuple[Path, Path, ShardReceipt]:
        """Close staging TAR, verify integrity & SHA-256, and atomically commit shard + receipt."""
        if self._finalized or self._tar is None:
            raise RuntimeError("Shard writer has already been finalized or aborted.")
        if not self._members:
            self.abort()
            raise ValueError(f"Cannot finalize empty shard '{self.shard_id}'.")

        self._tar.close()
        self._tar = None

        byte_size = self.staging_tar_path.stat().st_size
        sha256_hex = compute_file_sha256(self.staging_tar_path)

        ok, msg = verify_tar_shard_file(
            self.staging_tar_path,
            expected_sha256=sha256_hex,
            expected_byte_size=byte_size,
            expected_image_count=len(self._members),
        )
        if not ok:
            self.staging_tar_path.unlink(missing_ok=True)
            raise RuntimeError(f"Staging shard verification failed for '{self.shard_id}': {msg}")

        # Read exact on-disk byte offsets directly from the finalized TAR headers
        # (TarFile.addfile copies TarInfo internally, so mutating headers requires reading back).
        offset_by_name: dict[str, int] = {}
        with tarfile.open(self.staging_tar_path, mode="r:") as tf:
            for member in tf:
                if member.isfile():
                    offset_by_name[member.name] = int(member.offset_data)

        verified_members: list[ShardMemberEntry] = []
        for entry in self._members:
            if entry.member_name not in offset_by_name:
                self.staging_tar_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Member '{entry.member_name}' missing from finalized shard '{self.shard_id}'."
                )
            verified_members.append(
                ShardMemberEntry(
                    image_id=entry.image_id,
                    member_name=entry.member_name,
                    offset_data=offset_by_name[entry.member_name],
                    byte_size=entry.byte_size,
                    sha256=entry.sha256,
                    label=entry.label,
                    generator=entry.generator,
                    source_generator_group=entry.source_generator_group,
                    source_split=entry.source_split,
                    release_split=entry.release_split,
                    width=entry.width,
                    height=entry.height,
                    image_format=entry.image_format,
                )
            )
        self._members = verified_members

        # Atomic rename on local staging filesystem
        self.staging_tar_path.replace(self.final_tar_path)

        receipt = ShardReceipt(
            shard_id=self.shard_id,
            release_id=self.release_id,
            origin_release_id=self.release_id,
            release_split=self.release_split,
            storage_key=self.storage_key,
            status="COMMITTED",
            image_count=len(self._members),
            byte_size=byte_size,
            sha256=sha256_hex,
            created_at_utc=datetime.now(UTC).isoformat(),
            members=[m.to_dict() for m in self._members],
        )

        tmp_receipt = self.shard_dir / f"{self.shard_id}.receipt.json.tmp"
        tmp_receipt.write_text(json.dumps(receipt.to_dict(), indent=2), encoding="utf-8")
        tmp_receipt.replace(self.receipt_path)

        self._finalized = True
        return self.final_tar_path, self.receipt_path, receipt

    def abort(self) -> None:
        """Close and delete staging file on error or interruption."""
        if self._tar is not None:
            try:
                self._tar.close()
            finally:
                self._tar = None
        if self.staging_tar_path.exists():
            self.staging_tar_path.unlink(missing_ok=True)
        self._finalized = True
