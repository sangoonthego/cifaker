"""Production sharded Apache Arrow source adapter for `nebula/GenImage-arrow`.

Supports reading original encoded image bytes directly from pinned Hugging Face
Arrow IPC shards (`data/{source_split}/{generator_directory}/data-XXXXX-of-YYYYY.arrow`)
without re-encoding images, without loading complete Arrow shards into Pandas, and
without downloading unrelated upstream partitions.

Verified upstream schemas:
- Train Arrow schema:
    image_path : string
    md5        : string (optional)
    width      : int64  (optional)
    height     : int64  (optional)
    image      : binary
    label      : int64
    generator  : int64  (optional; never blindly converted to string for generator provenance)
- Test Arrow schema:
    image_path : string
    image      : binary
    label      : int64
    generator  : int64  (optional)

Canonical locator contract (preserved from frozen sampling snapshots):
    source_path = f"{shard_filename}/rows/{row_index:08d}/{normalized_original_path}"
with `source_shard`, `source_row_index`, and `original_image_path` validated for consistency.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as pa_ipc

from prooflens.data.config import SourceConfig
from prooflens.data.registry.schema import (
    LABEL_REAL,
    extract_imagenet_class_from_path,
    validate_safe_relative_path,
)
from prooflens.data.sources.base import (
    SourceAccessError,
    SourceAdapter,
    SourceCapabilities,
    SourceImagePayload,
    SourceRecord,
)

DEFAULT_ARROW_GENERATOR_MAPPING: dict[str, str] = {
    "ADM": "adm",
    "BigGAN": "biggan",
    "glide": "glide",
    "Midjourney": "midjourney",
    "stable_diffusion_v_1_4": "sdv4",
    "stable_diffusion_v_1_5": "sdv5",
    "VQDM": "vqdm",
    "wukong": "wukong",
}

_ARROW_SHARD_RE = re.compile(
    r"^data/(?P<split>[^/]+)/(?P<gen_dir>[^/]+)/(?P<filename>[^/]+\.arrow)$"
)
_ROW_SEGMENT_RE = re.compile(r"^(?P<shard>.+?)/rows/(?P<row_idx>-?\d+)/(?P<orig_path>.+)$")
_ARROW_FILE_MAGIC = b"ARROW1"


class ArrowSchemaError(SourceAccessError):
    """Raised when an upstream Arrow shard lacks mandatory columns or has incompatible types."""


class ArrowLocatorError(SourceAccessError):
    """Raised when an Arrow source shard, row index, or canonical locator is invalid or inconsistent."""


@dataclass(frozen=True)
class ParsedArrowShard:
    """Validated components of a GenImage-arrow shard path."""

    shard_path: str
    source_split: str
    generator_directory: str
    source_generator_group: str
    filename: str


@dataclass(frozen=True)
class ParsedArrowLocator:
    """Validated components of a canonical Arrow row locator."""

    shard_prefix: str
    row_index: int
    original_image_path: str


def normalize_original_image_path(raw_path: str) -> str:
    """Normalize an original image path inside an Arrow row and reject unsafe paths."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ArrowLocatorError("Original image_path in Arrow row must be a non-empty string.")
    cleaned = raw_path.strip().replace("\\", "/").lstrip("/")
    return validate_safe_relative_path(cleaned)


def build_arrow_source_path(
    shard_filename: str,
    row_index: int,
    original_image_path: str,
) -> str:
    """Construct canonical Arrow source_path matching the frozen sampling locator contract."""
    if row_index < 0:
        raise ArrowLocatorError(f"Row index must be non-negative, got {row_index}.")
    safe_shard = validate_safe_relative_path(shard_filename)
    safe_orig = normalize_original_image_path(original_image_path)
    return f"{safe_shard}/rows/{row_index:08d}/{safe_orig}"


def parse_arrow_shard_path(
    shard_path: str,
    split_mapping: dict[str, str] | None = None,
    generator_mapping: dict[str, str] | None = None,
    strict_generator_mapping: bool = True,
) -> ParsedArrowShard:
    """Validate and parse `data/{source_split}/{generator_directory}/data-XXXXX-of-YYYYY.arrow`."""
    safe_shard = validate_safe_relative_path(shard_path)
    match = _ARROW_SHARD_RE.match(safe_shard)
    if not match:
        raise ArrowLocatorError(
            f"Invalid Arrow shard path '{shard_path}': expected "
            "'data/<source_split>/<generator_directory>/<name>.arrow'."
        )

    raw_split = match.group("split")
    gen_dir = match.group("gen_dir")
    filename = match.group("filename")

    sp_map = (
        {k.lower(): str(v) for k, v in split_mapping.items()}
        if split_mapping
        else {"train": "train", "val": "val", "valid": "val", "test": "test"}
    )
    if raw_split.lower() not in sp_map:
        raise ArrowLocatorError(
            f"Unmapped source split '{raw_split}' in Arrow shard path '{safe_shard}'."
        )
    source_split = sp_map[raw_split.lower()]

    gen_map = dict(DEFAULT_ARROW_GENERATOR_MAPPING)
    if generator_mapping:
        gen_map.update(generator_mapping)
    gen_map_lower = {k.lower(): v for k, v in gen_map.items()}

    if gen_dir in gen_map:
        gen_group = gen_map[gen_dir]
    elif gen_dir.lower() in gen_map_lower:
        gen_group = gen_map_lower[gen_dir.lower()]
    elif strict_generator_mapping:
        raise ArrowLocatorError(
            f"Unmapped generator directory '{gen_dir}' in Arrow shard path '{safe_shard}'."
        )
    else:
        gen_group = gen_dir.lower()

    return ParsedArrowShard(
        shard_path=safe_shard,
        source_split=source_split,
        generator_directory=gen_dir,
        source_generator_group=gen_group,
        filename=filename,
    )


def parse_arrow_source_path(
    source_path: str,
    expected_source_shard: str | None = None,
    expected_row_index: int | None = None,
    expected_original_path: str | None = None,
) -> ParsedArrowLocator:
    """Parse and cross-validate canonical Arrow locator `source_path`."""
    safe_path = validate_safe_relative_path(source_path)
    match = _ROW_SEGMENT_RE.match(safe_path)
    if not match:
        raise ArrowLocatorError(
            f"Invalid Arrow canonical source_path '{source_path}': expected "
            "'<shard>/rows/<row_index>/<original_image_path>'."
        )

    shard_prefix = validate_safe_relative_path(match.group("shard"))
    row_idx = int(match.group("row_idx"))
    if row_idx < 0:
        raise ArrowLocatorError(f"Negative row index rejected in source_path: '{source_path}'.")
    orig_path = normalize_original_image_path(match.group("orig_path"))

    if expected_source_shard is not None:
        safe_expected_shard = validate_safe_relative_path(expected_source_shard)
        expected_basename = Path(safe_expected_shard).name
        if shard_prefix != safe_expected_shard and shard_prefix != expected_basename:
            raise ArrowLocatorError(
                f"Locator shard mismatch: source_path references '{shard_prefix}', "
                f"but record source_shard is '{safe_expected_shard}'."
            )

    if expected_row_index is not None:
        if expected_row_index < 0:
            raise ArrowLocatorError(f"Negative source_row_index rejected: {expected_row_index}.")
        if row_idx != expected_row_index:
            raise ArrowLocatorError(
                f"Locator row index mismatch: source_path has row {row_idx}, "
                f"but record source_row_index is {expected_row_index}."
            )

    if expected_original_path is not None:
        norm_expected_orig = normalize_original_image_path(expected_original_path)
        if orig_path != norm_expected_orig:
            raise ArrowLocatorError(
                f"Locator original_image_path mismatch: source_path has '{orig_path}', "
                f"but record original_image_path is '{norm_expected_orig}'."
            )

    return ParsedArrowLocator(
        shard_prefix=shard_prefix,
        row_index=row_idx,
        original_image_path=orig_path,
    )


def _validate_arrow_schema(schema: pa.Schema, shard_path: str) -> None:
    """Verify mandatory Arrow columns (`image_path`, `image`, `label`) and compatible types."""
    names = set(schema.names)
    mandatory = ("image_path", "image", "label")
    missing = [col for col in mandatory if col not in names]
    if missing:
        raise ArrowSchemaError(
            f"Arrow shard '{shard_path}' missing mandatory column(s): {missing}. "
            f"Available columns: {list(schema.names)}"
        )

    path_type = schema.field("image_path").type
    if not (pa.types.is_string(path_type) or pa.types.is_large_string(path_type)):
        raise ArrowSchemaError(
            f"Arrow shard '{shard_path}' column 'image_path' must be string, got {path_type}."
        )

    img_type = schema.field("image").type
    if not (
        pa.types.is_binary(img_type)
        or pa.types.is_large_binary(img_type)
        or pa.types.is_fixed_size_binary(img_type)
    ):
        raise ArrowSchemaError(
            f"Arrow shard '{shard_path}' column 'image' must be binary raw bytes, got {img_type}."
        )

    lbl_type = schema.field("label").type
    if not pa.types.is_integer(lbl_type):
        raise ArrowSchemaError(
            f"Arrow shard '{shard_path}' column 'label' must be integer, got {lbl_type}."
        )


def _is_arrow_ipc_file(path: Path) -> bool:
    """Inspect leading magic bytes to distinguish Arrow IPC File (`ARROW1`) from IPC Stream."""
    with path.open("rb") as handle:
        header = handle.read(len(_ARROW_FILE_MAGIC))
    return header == _ARROW_FILE_MAGIC


def iter_arrow_record_batches(
    local_arrow_path: Path,
    shard_label: str,
    columns: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream PyArrow RecordBatches with bounded memory supporting both IPC file and IPC stream."""
    if not local_arrow_path.is_file():
        raise FileNotFoundError(f"Arrow shard file not found: {local_arrow_path}")

    is_file = _is_arrow_ipc_file(local_arrow_path)
    with pa.memory_map(str(local_arrow_path), "r") as mm:
        if is_file:
            reader = pa_ipc.open_file(mm)
            _validate_arrow_schema(reader.schema, shard_label)
            avail = set(reader.schema.names)
            selected_cols = [c for c in columns if c in avail] if columns else None
            for batch_idx in range(reader.num_record_batches):
                batch = reader.get_batch(batch_idx)
                if selected_cols is not None:
                    batch = batch.select(selected_cols)
                yield batch
        else:
            stream_reader = pa_ipc.open_stream(mm)
            _validate_arrow_schema(stream_reader.schema, shard_label)
            avail = set(stream_reader.schema.names)
            selected_cols = [c for c in columns if c in avail] if columns else None
            for batch in stream_reader:
                if selected_cols is not None:
                    batch = batch.select(selected_cols)
                yield batch


def _resolve_hf_token(token_env_var: str = "HF_TOKEN") -> str | None:
    """Resolve optional Hugging Face token from environment variables without storing secrets."""
    for env_name in (token_env_var, "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(env_name)
        if val and val.strip():
            return val.strip()
    return None


class GenImageArrowAdapter(SourceAdapter):
    """Shard-aware Apache Arrow source adapter for `nebula/GenImage-arrow` and compatible repos.

    Supports:
    - Pinned Hugging Face dataset revision (`hf_hub_download` with retries and bounded LRU cache).
    - Optional local Arrow directory (`arrow_local_root`) or custom `shard_resolver` for offline tests.
    - Both Arrow IPC File and Arrow IPC Stream formats with bounded-memory RecordBatch streaming.
    - Train schema (with optional `md5`, `width`, `height`) and Test schema (without those columns).
    - Strict validation of canonical locators, labels, original paths, and frozen SHA-256 hashes.
    """

    def __init__(
        self,
        config: SourceConfig,
        shard_resolver: Callable[[str], Path] | None = None,
    ) -> None:
        self._config = config
        self._extensions = {e.lower() for e in config.supported_extensions}
        self._shard_resolver = shard_resolver
        self._local_root = (
            Path(config.arrow_local_root).resolve() if config.arrow_local_root else None
        )
        self._cache_dir = (
            Path(config.arrow_cache_dir).resolve()
            if config.arrow_cache_dir
            else Path("artifacts/arrow_cache").resolve()
        )
        self._max_cached_shards = max(1, int(config.arrow_max_cached_shards))
        self._max_cache_bytes = config.arrow_max_cache_bytes
        self._retries = max(1, int(config.arrow_download_retries))
        self._backoff = max(0.0, float(config.arrow_retry_backoff_seconds))

        # Bounded LRU cache tracking ONLY temporary shards managed by this adapter
        self._lru: OrderedDict[str, tuple[Path, int]] = OrderedDict()
        self._downloads = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._evictions = 0
        self._peak_cache_bytes = 0

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
            random_access=False,
            shard_level_access=True,
            resume_support=True,
            estimated_transfer_bytes=self._config.estimated_total_bytes,
            access_notes=(
                "Sharded Apache Arrow source on Hugging Face Hub (`nebula/GenImage-arrow`). "
                "Retrieving an image requires downloading or opening its containing `.arrow` shard; "
                "arbitrary remote single-row random access is NOT advertised. Use "
                "iter_selected_images() to group requests by source_shard with bounded LRU caching."
            ),
        )

    @property
    def current_cache_bytes(self) -> int:
        return sum(sz for _, sz in self._lru.values())

    def runtime_stats(self) -> dict[str, Any]:
        return {
            "downloads": self._downloads,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "evictions": self._evictions,
            "cached_shards_current": len(self._lru),
            "cached_bytes_current": self.current_cache_bytes,
            "peak_cache_bytes": self._peak_cache_bytes,
        }

    def _evict_if_needed(self, incoming_bytes: int = 0) -> None:
        while self._lru and (
            len(self._lru) >= self._max_cached_shards
            or (
                self._max_cache_bytes is not None
                and self.current_cache_bytes + incoming_bytes > self._max_cache_bytes
            )
        ):
            _old_shard, (old_path, _old_sz) = self._lru.popitem(last=False)
            old_path.unlink(missing_ok=True)
            self._evictions += 1

    def release_source_shard(self, source_shard: str) -> None:
        """Explicitly evict a managed temporary Arrow shard once no remaining tasks require it."""
        norm = validate_safe_relative_path(source_shard)
        entry = self._lru.pop(norm, None)
        if entry is not None:
            path, _ = entry
            path.unlink(missing_ok=True)
            self._evictions += 1

    def clear_cache(self) -> None:
        """Delete all managed temporary Arrow shards cached by this adapter."""
        for _shard, (path, _sz) in list(self._lru.items()):
            path.unlink(missing_ok=True)
            self._evictions += 1
        self._lru.clear()

    def _acquire_local_shard(self, source_shard: str) -> Path:
        """Resolve an Arrow shard locally or download it from HF Hub into the bounded LRU cache."""
        safe_shard = validate_safe_relative_path(source_shard)

        # 1. Check bounded LRU cache first
        if safe_shard in self._lru:
            cached_path, cached_sz = self._lru[safe_shard]
            if cached_path.is_file():
                self._lru.move_to_end(safe_shard)
                self._cache_hits += 1
                return cached_path
            self._lru.pop(safe_shard, None)

        # 2. Check configured read-only local mirror directory (offline tests or pre-staged files)
        if self._local_root is not None:
            candidate = (self._local_root / safe_shard).resolve()
            if not str(candidate).startswith(str(self._local_root)):
                raise ArrowLocatorError(
                    f"Shard path '{source_shard}' escapes arrow_local_root '{self._local_root}'."
                )
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Arrow shard not found under arrow_local_root: {candidate}"
                )
            self._cache_hits += 1
            return candidate

        # 3. Check custom injected shard resolver (used for unit testing download reuse & eviction)
        self._cache_misses += 1
        self._evict_if_needed()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        dest_path = (self._cache_dir / safe_shard).resolve()
        if not str(dest_path).startswith(str(self._cache_dir)):
            raise ArrowLocatorError(f"Shard path escapes arrow_cache_dir: {source_shard}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if self._shard_resolver is not None:
            last_exc: Exception | None = None
            for attempt in range(self._retries):
                try:
                    resolved = Path(self._shard_resolver(safe_shard))
                    if not resolved.is_file():
                        raise FileNotFoundError(f"Resolved Arrow shard missing: {resolved}")
                    if resolved.resolve() != dest_path:
                        shutil.copyfile(resolved, dest_path)
                    sz = dest_path.stat().st_size
                    self._downloads += 1
                    self._lru[safe_shard] = (dest_path, sz)
                    self._peak_cache_bytes = max(self._peak_cache_bytes, self.current_cache_bytes)
                    return dest_path
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 < self._retries and self._backoff > 0:
                        time.sleep(self._backoff * (2**attempt))
            raise SourceAccessError(
                f"SOURCE_FETCH_FAILURE: Failed to resolve Arrow shard '{safe_shard}' after "
                f"{self._retries} attempt(s): {last_exc}"
            ) from last_exc

        # 4. Production Hugging Face Hub pinned shard download
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SourceAccessError(
                "SOURCE_FETCH_FAILURE: GenImageArrowAdapter requires `huggingface-hub`. "
                "Install with `pip install 'prooflens[hf]'`."
            ) from exc

        token = _resolve_hf_token(self._config.hf_token_env_var)
        last_exc = None
        for attempt in range(self._retries):
            try:
                dl_path = Path(
                    hf_hub_download(
                        repo_id=self._config.location,
                        filename=safe_shard,
                        repo_type="dataset",
                        revision=self._config.source_revision,
                        local_dir=str(self._cache_dir),
                        token=token,
                    )
                )
                sz = dl_path.stat().st_size
                self._downloads += 1
                self._lru[safe_shard] = (dl_path, sz)
                self._peak_cache_bytes = max(self._peak_cache_bytes, self.current_cache_bytes)
                return dl_path
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self._retries and self._backoff > 0:
                    time.sleep(self._backoff * (2**attempt))

        raise SourceAccessError(
            f"SOURCE_FETCH_FAILURE: Failed to download Arrow shard '{safe_shard}' "
            f"from '{self._config.location}@{self._config.source_revision}' after "
            f"{self._retries} attempt(s): {last_exc}"
        ) from last_exc

    def list_available_shards(self) -> list[str]:
        """Discover available `data/<split>/<gen>/*.arrow` shards locally or via HF Hub API."""
        if self._local_root is not None:
            if not self._local_root.is_dir():
                raise FileNotFoundError(f"arrow_local_root does not exist: {self._local_root}")
            found: list[str] = []
            for p in sorted(self._local_root.rglob("*.arrow")):
                if not p.is_file():
                    continue
                rel = p.relative_to(self._local_root).as_posix()
                if _ARROW_SHARD_RE.match(rel):
                    found.append(rel)
            return found

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise SourceAccessError(
                "Listing remote HF Arrow shards requires `huggingface-hub` (`pip install 'prooflens[hf]'`)."
            ) from exc

        token = _resolve_hf_token(self._config.hf_token_env_var)
        api = HfApi(token=token)
        files = api.list_repo_files(
            repo_id=self._config.location,
            repo_type="dataset",
            revision=self._config.source_revision,
        )
        return sorted(f for f in files if _ARROW_SHARD_RE.match(f))

    def iter_records(self) -> Iterator[SourceRecord]:
        """Stream canonical SourceRecords across available Arrow shards with bounded memory."""
        for shard_rel in self.list_available_shards():
            parsed_shard = parse_arrow_shard_path(
                shard_rel,
                split_mapping=self._config.split_mapping,
                generator_mapping=self._config.generator_mapping,
                strict_generator_mapping=self._config.strict_generator_mapping,
            )
            local_path = self._acquire_local_shard(parsed_shard.shard_path)
            row_offset = 0
            try:
                for batch in iter_arrow_record_batches(
                    local_path,
                    shard_label=parsed_shard.shard_path,
                    columns=["image_path", "label", "image", "width", "height", "md5"],
                ):
                    paths_arr = batch.column("image_path")
                    labels_arr = batch.column("label")
                    images_arr = batch.column("image")
                    for local_i in range(batch.num_rows):
                        global_idx = row_offset + local_i
                        raw_orig_path = paths_arr[local_i].as_py()
                        orig_norm = normalize_original_image_path(str(raw_orig_path))
                        raw_label = labels_arr[local_i].as_py()
                        if raw_label not in (0, 1):
                            raise ArrowSchemaError(
                                f"Invalid binary label {raw_label!r} at {parsed_shard.shard_path} row {global_idx}."
                            )
                        label = int(raw_label)
                        raw_img = images_arr[local_i].as_py()
                        byte_len = len(raw_img) if isinstance(raw_img, (bytes, bytearray)) else None
                        ext = Path(orig_norm).suffix.lower()

                        gen_group = parsed_shard.source_generator_group
                        generator = (
                            self._config.real_generator_value if label == LABEL_REAL else gen_group
                        )
                        source_path = build_arrow_source_path(
                            parsed_shard.shard_path, global_idx, orig_norm
                        )

                        yield SourceRecord(
                            source_id=self.source_id,
                            source_revision=self.source_revision,
                            source_path=source_path,
                            source_split=parsed_shard.source_split,
                            label=label,
                            generator=generator,
                            source_generator_group=gen_group,
                            source_shard=parsed_shard.shard_path,
                            imagenet_class=extract_imagenet_class_from_path(orig_norm),
                            file_size_bytes=byte_len,
                            file_extension=ext,
                            source_row_index=global_idx,
                            original_image_path=orig_norm,
                        )
                    row_offset += batch.num_rows
            finally:
                self.release_source_shard(parsed_shard.shard_path)

    def fetch_image_bytes(self, record: SourceRecord) -> SourceImagePayload:
        """Single-record fetch is intentionally disabled to prevent one Arrow download per image."""
        raise SourceAccessError(
            "Single-image random access is disabled on GenImageArrowAdapter to prevent downloading "
            "a full Arrow shard per image. Use iter_selected_images() for shard-grouped retrieval."
        )

    def _validate_and_resolve_locator(self, rec: SourceRecord) -> tuple[str, int, str]:
        if rec.source_revision != self.source_revision:
            raise ArrowLocatorError(
                f"SOURCE_LOCATOR_MISMATCH: Source revision mismatch for '{rec.source_path}': "
                f"record has '{rec.source_revision}', adapter pinned to '{self.source_revision}'."
            )
        if not rec.source_shard:
            raise ArrowLocatorError(
                f"SOURCE_LOCATOR_MISMATCH: Missing source_shard on record '{rec.source_path}'."
            )

        parsed_shard = parse_arrow_shard_path(
            rec.source_shard,
            split_mapping=self._config.split_mapping,
            generator_mapping=self._config.generator_mapping,
            strict_generator_mapping=self._config.strict_generator_mapping,
        )
        parsed_loc = parse_arrow_source_path(
            rec.source_path,
            expected_source_shard=parsed_shard.shard_path,
            expected_row_index=rec.source_row_index,
            expected_original_path=rec.original_image_path,
        )
        return parsed_shard.shard_path, parsed_loc.row_index, parsed_loc.original_image_path

    def iter_selected_images(
        self,
        records: Iterable[SourceRecord],
    ) -> Iterator[tuple[SourceRecord, SourceImagePayload | None, str | None]]:
        """Group selected records by Arrow shard and retrieve exact original bytes in one pass per shard."""
        # Group validated requests by source_shard -> {row_index: (record, expected_orig_path)}
        grouped: dict[str, dict[int, tuple[SourceRecord, str]]] = defaultdict(dict)
        locator_errors: list[tuple[SourceRecord, str]] = []

        for rec in records:
            try:
                shard_path, row_idx, expected_orig = self._validate_and_resolve_locator(rec)
            except Exception as exc:
                locator_errors.append((rec, f"SOURCE_LOCATOR_MISMATCH: {exc}"))
                continue

            if row_idx in grouped[shard_path]:
                existing_rec, _ = grouped[shard_path][row_idx]
                locator_errors.append(
                    (
                        rec,
                        f"SOURCE_LOCATOR_MISMATCH: Duplicate Arrow locator ({shard_path}, row={row_idx}) "
                        f"collides with '{existing_rec.source_path}'.",
                    )
                )
                continue

            grouped[shard_path][row_idx] = (rec, expected_orig)

        for rec, err_msg in locator_errors:
            yield rec, None, err_msg

        for shard_path in sorted(grouped.keys()):
            wanted_rows = grouped[shard_path]
            try:
                local_arrow_path = self._acquire_local_shard(shard_path)
            except Exception as exc:
                for rec, _ in wanted_rows.values():
                    yield rec, None, f"SOURCE_FETCH_FAILURE: {type(exc).__name__}: {exc}"
                continue

            found_indices: set[int] = set()
            total_rows = 0
            try:
                for batch in iter_arrow_record_batches(
                    local_arrow_path,
                    shard_label=shard_path,
                    columns=["image_path", "image", "label"],
                ):
                    batch_start = total_rows
                    batch_end = total_rows + batch.num_rows
                    total_rows = batch_end

                    # Fast check: skip entire record batch if none of our wanted rows fall inside it
                    hits = [r_idx for r_idx in wanted_rows if batch_start <= r_idx < batch_end]
                    if not hits:
                        continue

                    paths_col = batch.column("image_path")
                    images_col = batch.column("image")
                    labels_col = batch.column("label")

                    for r_idx in sorted(hits):
                        found_indices.add(r_idx)
                        rec, expected_orig = wanted_rows[r_idx]
                        local_i = r_idx - batch_start

                        # Validate original path match
                        raw_arrow_path = paths_col[local_i].as_py()
                        try:
                            norm_arrow_path = normalize_original_image_path(str(raw_arrow_path))
                        except Exception as exc:
                            yield (
                                rec,
                                None,
                                f"SOURCE_LOCATOR_MISMATCH: Invalid Arrow image_path: {exc}",
                            )
                            continue

                        if norm_arrow_path != expected_orig:
                            yield (
                                rec,
                                None,
                                f"SOURCE_LOCATOR_MISMATCH: Arrow row {r_idx} in '{shard_path}' has "
                                f"image_path='{norm_arrow_path}', expected '{expected_orig}'.",
                            )
                            continue

                        # Validate label match
                        arrow_label = labels_col[local_i].as_py()
                        if arrow_label not in (0, 1) or int(arrow_label) != int(rec.label):
                            yield (
                                rec,
                                None,
                                f"SOURCE_LOCATOR_MISMATCH: Arrow row {r_idx} in '{shard_path}' has "
                                f"label={arrow_label!r}, expected {rec.label}.",
                            )
                            continue

                        # Extract raw original bytes verbatim (never re-encode!)
                        raw_val = images_col[local_i].as_py()
                        if not isinstance(raw_val, (bytes, bytearray, memoryview)):
                            yield (
                                rec,
                                None,
                                f"INVALID_IMAGE_PAYLOAD: Arrow row {r_idx} in '{shard_path}' "
                                f"returned non-binary payload type {type(raw_val).__name__}.",
                            )
                            continue

                        raw_bytes = bytes(raw_val)
                        if len(raw_bytes) == 0:
                            yield (
                                rec,
                                None,
                                f"INVALID_IMAGE_PAYLOAD: Empty image bytes at row {r_idx} in '{shard_path}'.",
                            )
                            continue

                        # Validate exact byte size when frozen metadata provides it
                        if rec.file_size_bytes is not None and len(raw_bytes) != int(
                            rec.file_size_bytes
                        ):
                            yield (
                                rec,
                                None,
                                f"SHA256_MISMATCH: File size mismatch at row {r_idx} in '{shard_path}': "
                                f"expected {rec.file_size_bytes} bytes, got {len(raw_bytes)}.",
                            )
                            continue

                        # Validate exact frozen SHA-256 when provided
                        if rec.expected_sha256:
                            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
                            if actual_sha != rec.expected_sha256:
                                yield (
                                    rec,
                                    None,
                                    f"SHA256_MISMATCH: Frozen SHA-256 mismatch at row {r_idx} "
                                    f"in '{shard_path}': expected {rec.expected_sha256}, got {actual_sha}.",
                                )
                                continue

                        ext = rec.file_extension or Path(norm_arrow_path).suffix.lower()
                        yield (
                            rec,
                            SourceImagePayload(
                                source_path=rec.source_path,
                                raw_bytes=raw_bytes,
                                file_extension=ext,
                                source_shard=shard_path,
                            ),
                            None,
                        )

                    if len(found_indices) == len(wanted_rows):
                        break
            except Exception as exc:
                for r_idx in sorted(set(wanted_rows.keys()) - found_indices):
                    rec, _ = wanted_rows[r_idx]
                    yield rec, None, f"SOURCE_FETCH_FAILURE: {type(exc).__name__}: {exc}"
                continue

            # Report out-of-range row indices cleanly
            for missing_idx in sorted(set(wanted_rows.keys()) - found_indices):
                rec, _ = wanted_rows[missing_idx]
                yield (
                    rec,
                    None,
                    f"SOURCE_LOCATOR_MISMATCH: Row index {missing_idx} out of range for Arrow shard "
                    f"'{shard_path}' (total_rows={total_rows}).",
                )
