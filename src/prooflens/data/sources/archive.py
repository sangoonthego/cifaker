"""Archive-based GenImage source adapter (ZIP, TAR, TAR.GZ) with shard-aware streaming.

Handles either a single archive file or a directory containing multiple generator
archives (the standard Full GenImage distribution layout).

Key scalability property:
- When reading selected records via `iter_selected_images`, requests are grouped
  by `source_shard` (archive path) so each archive is opened and streamed at most
  once, avoiding O(N) archive re-reads or per-image random-access emulation.
"""

from __future__ import annotations

import tarfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from prooflens.data.config import SourceConfig
from prooflens.data.registry.schema import (
    LABEL_REAL,
    extract_imagenet_class_from_path,
    validate_safe_relative_path,
)
from prooflens.data.scanner import parse_relative_image_path
from prooflens.data.sources.base import (
    SourceAccessError,
    SourceAdapter,
    SourceCapabilities,
    SourceImagePayload,
    SourceRecord,
)

ARCHIVE_SUFFIXES: tuple[str, ...] = (".zip", ".tar", ".tar.gz", ".tgz")


def _is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suf) for suf in ARCHIVE_SUFFIXES)


def _archive_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        return name[:-7]
    return path.stem


class ArchiveSourceAdapter(SourceAdapter):
    """SourceAdapter reading images from one or more ZIP / TAR / TAR.GZ archives."""

    def __init__(self, config: SourceConfig) -> None:
        self._config = config
        self._location = Path(config.location)
        self._extensions = {e.lower() for e in config.supported_extensions}
        self._archives = self._discover_archives()

    def _discover_archives(self) -> list[Path]:
        if self._location.is_file():
            if not _is_supported_archive(self._location):
                raise SourceAccessError(f"Unsupported archive file: {self._location}")
            return [self._location]
        if not self._location.is_dir():
            raise FileNotFoundError(f"Archive source location does not exist: {self._location}")

        pattern = self._config.archive_glob or "*"
        found = [
            p
            for p in sorted(self._location.rglob(pattern))
            if p.is_file() and _is_supported_archive(p)
        ]
        return found

    @property
    def source_id(self) -> str:
        return self._config.source_id

    @property
    def source_revision(self) -> str:
        return self._config.source_revision

    def capabilities(self) -> SourceCapabilities:
        all_zip = bool(self._archives) and all(p.suffix.lower() == ".zip" for p in self._archives)
        all_uncompressed_tar = bool(self._archives) and all(
            p.suffix.lower() == ".tar" for p in self._archives
        )
        total_archive_bytes = sum(p.stat().st_size for p in self._archives if p.is_file())
        est_bytes = (
            self._config.estimated_total_bytes
            if self._config.estimated_total_bytes is not None
            else (total_archive_bytes if total_archive_bytes > 0 else None)
        )
        return SourceCapabilities(
            metadata_only_enumeration=all_zip or all_uncompressed_tar,
            sequential_streaming=True,
            random_access=all_zip,
            shard_level_access=True,
            resume_support=True,
            estimated_transfer_bytes=est_bytes,
            access_notes=(
                "ZIP archives support central-directory metadata enumeration and member lookup. "
                "TAR/TAR.GZ archives require sequential shard-level streaming; iter_selected_images "
                "groups reads per archive shard to avoid repeated full scans."
            ),
        )

    def _parse_member_to_record(
        self,
        archive_path: Path,
        member_name: str,
        file_size: int | None,
    ) -> SourceRecord | None:
        norm_member = member_name.replace("\\", "/").lstrip("/")
        if not norm_member or norm_member.endswith("/"):
            return None

        ext = Path(norm_member).suffix.lower()
        if ext not in self._extensions:
            return None

        # Prepend archive stem if the member path only has <split>/<class>/<file>
        parts = [p for p in norm_member.split("/") if p and p != "."]
        parse_target = norm_member
        if len(parts) == 3:
            parse_target = f"{_archive_stem(archive_path)}/{norm_member}"

        parsed = parse_relative_image_path(
            relative_path=parse_target,
            label_mapping=self._config.label_mapping,
            split_mapping=self._config.split_mapping,
            generator_mapping=self._config.generator_mapping,
            strict_generator_mapping=self._config.strict_generator_mapping,
        )

        shard_rel = (
            archive_path.name
            if self._location.is_file()
            else archive_path.relative_to(self._location).as_posix()
        )
        source_path = validate_safe_relative_path(f"{shard_rel}::{norm_member}")

        # Enforce REAL provenance rule: REAL images never attribute a generative model in `generator`
        gen_group = parsed.generator
        generator = self._config.real_generator_value if parsed.label == LABEL_REAL else gen_group

        return SourceRecord(
            source_id=self.source_id,
            source_revision=self.source_revision,
            source_path=source_path,
            source_split=parsed.source_split,
            label=parsed.label,
            generator=generator,
            source_generator_group=gen_group,
            source_shard=shard_rel,
            imagenet_class=extract_imagenet_class_from_path(norm_member),
            file_size_bytes=file_size,
            file_extension=ext,
        )

    def _resolve_archive_path(self, source_shard: str) -> Path:
        if self._location.is_file():
            return self._location
        candidate = (self._location / source_shard).resolve()
        if not str(candidate).startswith(str(self._location.resolve())):
            raise SourceAccessError(f"Unsafe shard path outside archive root: {source_shard}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Archive shard not found: {candidate}")
        return candidate

    def iter_records(self) -> Iterator[SourceRecord]:
        for archive_path in self._archives:
            lower_name = archive_path.name.lower()
            if lower_name.endswith(".zip"):
                with zipfile.ZipFile(archive_path, mode="r") as zf:
                    for info in sorted(zf.infolist(), key=lambda i: i.filename):
                        if info.is_dir():
                            continue
                        rec = self._parse_member_to_record(
                            archive_path, info.filename, info.file_size
                        )
                        if rec is not None:
                            yield rec
            else:
                mode = "r:gz" if lower_name.endswith((".tar.gz", ".tgz")) else "r:"
                with tarfile.open(archive_path, mode=mode) as tf:
                    for member in tf:
                        if not member.isfile():
                            continue
                        rec = self._parse_member_to_record(archive_path, member.name, member.size)
                        if rec is not None:
                            yield rec

    def _split_source_path(self, record: SourceRecord) -> tuple[str, str]:
        if "::" not in record.source_path:
            raise SourceAccessError(
                f"Archive source_path must contain '<shard>::<member>', got: {record.source_path}"
            )
        shard_rel, member_name = record.source_path.split("::", 1)
        return shard_rel, member_name

    def fetch_image_bytes(self, record: SourceRecord) -> SourceImagePayload:
        shard_rel, member_name = self._split_source_path(record)
        archive_path = self._resolve_archive_path(shard_rel)
        lower_name = archive_path.name.lower()

        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, mode="r") as zf:
                raw_bytes = zf.read(member_name)
            return SourceImagePayload(
                source_path=record.source_path,
                raw_bytes=raw_bytes,
                file_extension=record.file_extension or Path(member_name).suffix.lower(),
                source_shard=shard_rel,
            )

        raise SourceAccessError(
            f"Single-image random access is disabled for sequential TAR archive '{shard_rel}'. "
            "Use iter_selected_images() for shard-aware streaming reads."
        )

    def iter_selected_images(
        self,
        records: Iterable[SourceRecord],
    ) -> Iterator[tuple[SourceRecord, SourceImagePayload | None, str | None]]:
        """Group selected records by archive shard and read each archive in a single pass."""
        grouped: dict[str, dict[str, SourceRecord]] = defaultdict(dict)
        for rec in records:
            shard_rel, member_name = self._split_source_path(rec)
            grouped[shard_rel][member_name] = rec

        for shard_rel in sorted(grouped.keys()):
            wanted = grouped[shard_rel]
            try:
                archive_path = self._resolve_archive_path(shard_rel)
            except Exception as exc:
                for rec in wanted.values():
                    yield rec, None, f"{type(exc).__name__}: {exc}"
                continue

            lower_name = archive_path.name.lower()
            if lower_name.endswith(".zip"):
                try:
                    with zipfile.ZipFile(archive_path, mode="r") as zf:
                        for member_name in sorted(wanted.keys()):
                            rec = wanted[member_name]
                            try:
                                raw = zf.read(member_name)
                                payload = SourceImagePayload(
                                    source_path=rec.source_path,
                                    raw_bytes=raw,
                                    file_extension=rec.file_extension
                                    or Path(member_name).suffix.lower(),
                                    source_shard=shard_rel,
                                )
                                yield rec, payload, None
                            except Exception as exc:
                                yield rec, None, f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    for rec in wanted.values():
                        yield rec, None, f"{type(exc).__name__}: {exc}"
            else:
                mode = "r:gz" if lower_name.endswith((".tar.gz", ".tgz")) else "r:"
                found_members: set[str] = set()
                try:
                    with tarfile.open(archive_path, mode=mode) as tf:
                        for member in tf:
                            if not member.isfile():
                                continue
                            norm_name = member.name.replace("\\", "/").lstrip("/")
                            if norm_name in wanted:
                                rec = wanted[norm_name]
                                found_members.add(norm_name)
                                try:
                                    extracted = tf.extractfile(member)
                                    if extracted is None:
                                        raise SourceAccessError(
                                            f"Unable to extract TAR member: {norm_name}"
                                        )
                                    raw = extracted.read()
                                    payload = SourceImagePayload(
                                        source_path=rec.source_path,
                                        raw_bytes=raw,
                                        file_extension=rec.file_extension
                                        or Path(norm_name).suffix.lower(),
                                        source_shard=shard_rel,
                                    )
                                    yield rec, payload, None
                                except Exception as exc:
                                    yield rec, None, f"{type(exc).__name__}: {exc}"
                                if len(found_members) == len(wanted):
                                    break
                except Exception as exc:
                    for member_name, rec in wanted.items():
                        if member_name not in found_members:
                            yield rec, None, f"{type(exc).__name__}: {exc}"
                    continue

                for member_name in sorted(set(wanted.keys()) - found_members):
                    yield wanted[member_name], None, f"Missing TAR member: {member_name}"
