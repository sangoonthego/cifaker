"""GenImage directory and Hugging Face source adapters + adapter factory.

1. `DirectoryGenImageAdapter`:
   Reads extracted GenImage / Tiny GenImage directory hierarchies on local disk,
   Colab runtime storage, or mounted volumes. Supports metadata-only enumeration
   without decoding images, O(1) random file access, and strict REAL provenance
   separation (`generator="none"` vs `source_generator_group=<subset>`).

2. `HuggingFaceGenImageAdapter`:
   Explicitly configured Hugging Face dataset adapter requiring a pinned
   `source_revision` and explicit `hf_column_mapping`. Does NOT hardcode an
   unverified community repository schema. Declares honest streaming capabilities.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from prooflens.data.config import SourceConfig
from prooflens.data.registry.schema import (
    LABEL_REAL,
    extract_imagenet_class_from_path,
    validate_safe_relative_path,
)
from prooflens.data.scanner import parse_relative_image_path
from prooflens.data.sources.archive import ArchiveSourceAdapter
from prooflens.data.sources.base import (
    SourceAccessError,
    SourceAdapter,
    SourceCapabilities,
    SourceImagePayload,
    SourceRecord,
)

_IGNORED_FILENAMES: frozenset[str] = frozenset(
    {".gitkeep", ".DS_Store", "Thumbs.db", "desktop.ini", ".acquisition_metadata.json"}
)


class DirectoryGenImageAdapter(SourceAdapter):
    """SourceAdapter for extracted GenImage directory trees."""

    def __init__(self, config: SourceConfig) -> None:
        self._config = config
        self._root = Path(config.location)
        self._extensions = {e.lower() for e in config.supported_extensions}

    @property
    def source_id(self) -> str:
        return self._config.source_id

    @property
    def source_revision(self) -> str:
        return self._config.source_revision

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            metadata_only_enumeration=True,
            sequential_streaming=True,
            random_access=True,
            shard_level_access=False,
            resume_support=True,
            estimated_transfer_bytes=self._config.estimated_total_bytes,
            access_notes=(
                "Directory-backed GenImage source supports metadata-only os.walk enumeration "
                "and direct random-access file reads without unpacking archives."
            ),
        )

    def iter_records(self) -> Iterator[SourceRecord]:
        if not self._root.is_dir():
            raise FileNotFoundError(f"Directory source root does not exist: {self._root}")

        for dirpath, dirnames, filenames in os.walk(self._root, followlinks=True):
            dirnames.sort()
            for fname in sorted(filenames):
                if fname in _IGNORED_FILENAMES or fname.startswith("."):
                    continue
                abs_path = Path(dirpath) / fname
                ext = abs_path.suffix.lower()
                if ext not in self._extensions:
                    continue

                rel_posix = validate_safe_relative_path(abs_path.relative_to(self._root).as_posix())
                parsed = parse_relative_image_path(
                    relative_path=rel_posix,
                    label_mapping=self._config.label_mapping,
                    split_mapping=self._config.split_mapping,
                    generator_mapping=self._config.generator_mapping,
                    strict_generator_mapping=self._config.strict_generator_mapping,
                )

                gen_group = parsed.generator
                generator = (
                    self._config.real_generator_value if parsed.label == LABEL_REAL else gen_group
                )

                try:
                    file_size = abs_path.stat().st_size
                except OSError:
                    file_size = None

                # Treat top-level directory as logical source_shard for locality grouping
                top_dir = rel_posix.split("/", 1)[0]

                yield SourceRecord(
                    source_id=self.source_id,
                    source_revision=self.source_revision,
                    source_path=rel_posix,
                    source_split=parsed.source_split,
                    label=parsed.label,
                    generator=generator,
                    source_generator_group=gen_group,
                    source_shard=top_dir,
                    imagenet_class=extract_imagenet_class_from_path(rel_posix),
                    file_size_bytes=file_size,
                    file_extension=ext,
                )

    def fetch_image_bytes(self, record: SourceRecord) -> SourceImagePayload:
        safe_rel = validate_safe_relative_path(record.source_path)
        root_resolved = self._root.resolve()
        target = (root_resolved / safe_rel).resolve()
        if not str(target).startswith(str(root_resolved)):
            raise SourceAccessError(f"Path traversal blocked for source_path: {record.source_path}")
        if not target.is_file():
            raise FileNotFoundError(f"Source image file not found: {target}")

        raw_bytes = target.read_bytes()
        return SourceImagePayload(
            source_path=safe_rel,
            raw_bytes=raw_bytes,
            file_extension=record.file_extension or target.suffix.lower(),
            source_shard=record.source_shard,
        )


class HuggingFaceGenImageAdapter(SourceAdapter):
    """Configurable Hugging Face dataset adapter with explicit schema mapping and revision pinning.

    Important operational constraints:
    - Does NOT assume any hardcoded community mirror layout; callers must specify
      `location` (HF repo ID), `source_revision` (pinned commit SHA/tag), and
      `hf_column_mapping` (`image`, `label`, `generator_group`, `split`, `path`).
    - When `hf_streaming=True`, metadata-only enumeration without reading image payloads
      is NOT supported (`metadata_only_enumeration=False`) and random access is unavailable;
      materialization must use sequential streaming via `iter_selected_images()`.
    """

    def __init__(self, config: SourceConfig) -> None:
        self._config = config
        required_map_keys = {"image", "label", "generator_group"}
        missing = required_map_keys - set(config.hf_column_mapping.keys())
        if missing:
            raise ValueError(
                f"HuggingFaceGenImageAdapter requires explicit 'hf_column_mapping' keys "
                f"{sorted(required_map_keys)}, missing: {sorted(missing)}"
            )

    @property
    def source_id(self) -> str:
        return self._config.source_id

    @property
    def source_revision(self) -> str:
        return self._config.source_revision

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            metadata_only_enumeration=False,
            sequential_streaming=True,
            random_access=not self._config.hf_streaming,
            shard_level_access=True,
            resume_support=True,
            estimated_transfer_bytes=self._config.estimated_total_bytes,
            access_notes=(
                "Hugging Face streaming requires traversing dataset rows containing embedded "
                "image bytes; metadata-only inventory incurs streaming transfer cost. "
                "Pin source_revision to an immutable HF commit SHA before building releases."
            ),
        )

    def _load_hf_dataset_dict(self) -> dict[str, Any]:
        try:
            from datasets import Image as HfImage
            from datasets import load_dataset
        except ImportError as exc:
            raise SourceAccessError(
                "Hugging Face adapter requires optional dependencies: `pip install 'prooflens[hf]'`."
            ) from exc

        ds_dict = load_dataset(
            self._config.location,
            name=self._config.hf_config_name,
            revision=self._config.source_revision,
            streaming=self._config.hf_streaming,
        )
        img_col = self._config.hf_column_mapping["image"]
        result: dict[str, Any] = {}
        if isinstance(ds_dict, dict):
            for split_name, ds in ds_dict.items():
                # Keep original encoded bytes without decoding eagerly
                result[split_name] = ds.cast_column(img_col, HfImage(decode=False))
        else:
            result["train"] = ds_dict.cast_column(img_col, HfImage(decode=False))
        return result

    def _row_to_record_and_bytes(
        self,
        split_name: str,
        row_idx: int,
        row: dict[str, Any],
    ) -> tuple[SourceRecord, bytes | None]:
        cmap = self._config.hf_column_mapping
        raw_lbl = row[cmap["label"]]
        if isinstance(raw_lbl, int) and raw_lbl in (0, 1):
            label = raw_lbl
        else:
            lbl_key = str(raw_lbl).strip().lower()
            if lbl_key not in self._config.label_mapping:
                raise SourceAccessError(f"Unmapped Hugging Face label value: {raw_lbl!r}")
            label = self._config.label_mapping[lbl_key]

        raw_gen = str(row[cmap["generator_group"]]).strip()
        gen_group = self._config.generator_mapping.get(raw_gen, raw_gen)
        generator = self._config.real_generator_value if label == LABEL_REAL else gen_group

        split_key = str(row.get(cmap.get("split", ""), split_name)).strip().lower()
        source_split = self._config.split_mapping.get(split_key, split_key)

        img_val = row[cmap["image"]]
        raw_bytes: bytes | None = None
        img_path_hint: str | None = None
        if isinstance(img_val, dict):
            raw_bytes = img_val.get("bytes")
            img_path_hint = img_val.get("path")
        elif hasattr(img_val, "save"):
            buf = io.BytesIO()
            fmt = getattr(img_val, "format", None) or "PNG"
            img_val.save(buf, format=fmt)
            raw_bytes = buf.getvalue()

        path_col = cmap.get("path")
        if path_col and row.get(path_col):
            raw_path = str(row[path_col])
        elif img_path_hint:
            raw_path = f"{gen_group}/{source_split}/{Path(img_path_hint).name}"
        else:
            raw_path = f"{gen_group}/{source_split}/row_{row_idx:08d}.png"

        safe_path = validate_safe_relative_path(raw_path)
        ext = Path(safe_path).suffix.lower() or ".png"

        rec = SourceRecord(
            source_id=self.source_id,
            source_revision=self.source_revision,
            source_path=safe_path,
            source_split=source_split,
            label=label,
            generator=generator,
            source_generator_group=gen_group,
            source_shard=f"hf_{split_name}",
            imagenet_class=extract_imagenet_class_from_path(safe_path),
            file_size_bytes=len(raw_bytes) if raw_bytes is not None else None,
            file_extension=ext,
        )
        return rec, raw_bytes

    def iter_records(self) -> Iterator[SourceRecord]:
        splits = self._load_hf_dataset_dict()
        for split_name in sorted(splits.keys()):
            for idx, row in enumerate(splits[split_name]):
                rec, _ = self._row_to_record_and_bytes(split_name, idx, row)
                yield rec

    def fetch_image_bytes(self, record: SourceRecord) -> SourceImagePayload:
        raise SourceAccessError(
            "Single-image random access is not supported on streaming Hugging Face datasets. "
            "Use iter_selected_images() to stream selected image records in a single pass."
        )

    def iter_selected_images(
        self,
        records: Iterable[SourceRecord],
    ) -> Iterator[tuple[SourceRecord, SourceImagePayload | None, str | None]]:
        wanted_by_path = {rec.source_path: rec for rec in records}
        found: set[str] = set()
        splits = self._load_hf_dataset_dict()

        for split_name in sorted(splits.keys()):
            for idx, row in enumerate(splits[split_name]):
                rec, raw_bytes = self._row_to_record_and_bytes(split_name, idx, row)
                if rec.source_path in wanted_by_path:
                    orig_rec = wanted_by_path[rec.source_path]
                    found.add(rec.source_path)
                    if raw_bytes is None:
                        yield orig_rec, None, "Empty image payload returned by Hugging Face row."
                    else:
                        yield (
                            orig_rec,
                            SourceImagePayload(
                                source_path=orig_rec.source_path,
                                raw_bytes=raw_bytes,
                                file_extension=orig_rec.file_extension,
                                source_shard=orig_rec.source_shard,
                            ),
                            None,
                        )
                    if len(found) == len(wanted_by_path):
                        return

        for missing_path in sorted(set(wanted_by_path.keys()) - found):
            yield (
                wanted_by_path[missing_path],
                None,
                f"Record not encountered during Hugging Face stream: {missing_path}",
            )


def create_source_adapter(config: SourceConfig) -> SourceAdapter:
    """Instantiate the appropriate SourceAdapter from a validated SourceConfig."""
    if config.source_type == "directory":
        return DirectoryGenImageAdapter(config)
    if config.source_type == "archive":
        return ArchiveSourceAdapter(config)
    if config.source_type == "huggingface":
        return HuggingFaceGenImageAdapter(config)
    raise ValueError(f"Unsupported source_type: {config.source_type!r}")
