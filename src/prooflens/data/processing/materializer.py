"""Bounded-memory incremental image materialization engine with resume & parent-shard reuse.

Features:
- Reuses verified parent release shards without re-materializing or duplicating existing images.
- Assigns newly selected (delta) samples deterministically into fixed-size or count-bounded shards.
- Streams/batches image reads respecting source capabilities (parallel fetch with bounded
  in-flight window for random-access sources; shard-aware single-pass streaming for Arrow/archives/HF).
- Releases finished upstream Arrow source shards proactively once no remaining output shards need them.
- Retries transient source read errors with exponential backoff.
- Validates every image's raw bytes, dimensions, format, frozen SHA-256 (when present), and
  optional pixel/perceptual hash without altering original encoded bytes.
- Distinguishes explicit failure categories (`SOURCE_FETCH_FAILURE`, `SOURCE_LOCATOR_MISMATCH`,
  `SHA256_MISMATCH`, `INVALID_IMAGE_PAYLOAD`, `IMAGE_DECODE_FAILURE`, `STORAGE_FAILURE`).
- Supports `strict_frozen_mode=True` to fail immediately on any frozen record failure (never silently
  skipping or substituting alternative records).
- Resumes cleanly by verifying existing remote/local shard receipts + SHA-256 checksums and
  guarding against duplicate image commits across shards.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from prooflens.data.config import ReleaseConfig
from prooflens.data.processing.shard_writer import (
    ImmutableTarShardWriter,
    ShardReceipt,
    verify_tar_shard_file,
)
from prooflens.data.processing.validator import ByteValidationResult, validate_image_bytes
from prooflens.data.registry.duplicates import DuplicateAuditReport, assign_leakage_groups
from prooflens.data.sources.base import SourceAdapter, SourceImagePayload, SourceRecord
from prooflens.data.storage.base import StorageBackend

_FORMAT_TO_EXTENSION: dict[str, str] = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}

_FORMAT_EQUIVALENT_EXTS: dict[str, set[str]] = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
    "BMP": {".bmp"},
    "TIFF": {".tif", ".tiff"},
}


class MaterializationFailureCategory(StrEnum):
    """Explicit failure classification for materialization errors."""

    SOURCE_FETCH_FAILURE = "SOURCE_FETCH_FAILURE"
    SOURCE_LOCATOR_MISMATCH = "SOURCE_LOCATOR_MISMATCH"
    SHA256_MISMATCH = "SHA256_MISMATCH"
    INVALID_IMAGE_PAYLOAD = "INVALID_IMAGE_PAYLOAD"
    IMAGE_DECODE_FAILURE = "IMAGE_DECODE_FAILURE"
    STORAGE_FAILURE = "STORAGE_FAILURE"


class FrozenMaterializationError(RuntimeError):
    """Raised in `strict_frozen_mode` when any frozen selected record cannot be materialized."""

    def __init__(
        self,
        message: str,
        image_id: str,
        category: MaterializationFailureCategory,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.image_id = image_id
        self.category = category
        self.details = details or {}


@dataclass(frozen=True)
class PlannedShardSpec:
    """Deterministic membership specification for a new TAR shard to materialize."""

    shard_id: str
    release_split: str
    storage_key: str
    receipt_storage_key: str
    image_ids: tuple[str, ...]
    source_shards: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterializationResult:
    """Outcome returned by `materialize_release`."""

    release_id: str
    manifest_df: pd.DataFrame
    receipts: list[ShardReceipt]
    reused_parent_shards: int
    resumed_existing_shards: int
    newly_materialized_shards: int
    excluded_records: list[dict[str, Any]]
    duplicate_report: DuplicateAuditReport
    duration_seconds: float
    metrics: dict[str, Any]


def _make_deterministic_shard_id(
    release_id: str,
    release_split: str,
    shard_index: int,
    image_ids: list[str],
) -> str:
    digest = hashlib.sha256(
        f"{release_id}|{release_split}|{'|'.join(image_ids)}".encode()
    ).hexdigest()[:10]
    return f"{release_split}-{shard_index:05d}-{digest}"


def _classify_error_message(err_msg: str) -> MaterializationFailureCategory:
    upper = err_msg.upper()
    if upper.startswith("SHA256_MISMATCH") or "SHA-256 MISMATCH" in upper or "SHA256" in upper:
        return MaterializationFailureCategory.SHA256_MISMATCH
    if upper.startswith("SOURCE_LOCATOR_MISMATCH") or "LOCATOR" in upper:
        return MaterializationFailureCategory.SOURCE_LOCATOR_MISMATCH
    if (
        upper.startswith("INVALID_IMAGE_PAYLOAD")
        or upper.startswith("EMPTY_FILE")
        or upper.startswith("UNSUPPORTED_EXTENSION")
        or upper.startswith("INVALID_DIMENSIONS")
    ):
        return MaterializationFailureCategory.INVALID_IMAGE_PAYLOAD
    if upper.startswith("CORRUPTED:") or "DECODE" in upper:
        return MaterializationFailureCategory.IMAGE_DECODE_FAILURE
    if upper.startswith("STORAGE_FAILURE"):
        return MaterializationFailureCategory.STORAGE_FAILURE
    return MaterializationFailureCategory.SOURCE_FETCH_FAILURE


def _resolve_output_extension(original_ext: str, detected_format: str | None) -> str:
    """Choose output TAR member extension using actual image format when original ext is missing/inconsistent."""
    clean_orig = original_ext.lower().strip()
    if clean_orig and not clean_orig.startswith("."):
        clean_orig = f".{clean_orig}"
    fmt = (detected_format or "").upper().strip()
    if fmt in _FORMAT_EQUIVALENT_EXTS:
        if clean_orig in _FORMAT_EQUIVALENT_EXTS[fmt]:
            return clean_orig
        return _FORMAT_TO_EXTENSION[fmt]
    return clean_orig or ".bin"


def plan_deterministic_shards(
    delta_df: pd.DataFrame,
    release_id: str,
    shard_target_bytes: int,
    max_samples_per_shard: int | None = None,
) -> list[PlannedShardSpec]:
    """Partition delta rows deterministically into bounded-size TAR shard specs per split.

    Sorting by `(release_split, source_shard, source_row_index, image_id)` groups
    upstream Arrow/archive rows together to maximize source shard locality while keeping
    output shard membership 100% deterministic.
    """
    if delta_df.empty:
        return []

    specs: list[PlannedShardSpec] = []
    sort_cols = ["release_split"]
    if "source_shard" in delta_df.columns:
        sort_cols.append("source_shard")
    if "source_row_index" in delta_df.columns and delta_df["source_row_index"].notna().any():
        sort_cols.append("source_row_index")
    sort_cols.append("image_id")

    for split_name, grp in delta_df.groupby("release_split", sort=True):
        ordered = grp.sort_values(by=sort_cols, kind="mergesort")
        current_ids: list[str] = []
        current_src_shards: set[str] = set()
        current_est_bytes = 0
        shard_idx = 0

        for _, row in ordered.iterrows():
            img_id = str(row["image_id"])
            raw_sz = row.get("file_size_bytes")
            est_sz = int(raw_sz) if pd.notna(raw_sz) and int(raw_sz) > 0 else 256 * 1024

            hit_bytes = current_ids and (current_est_bytes + est_sz > shard_target_bytes)
            hit_count = (
                max_samples_per_shard is not None and len(current_ids) >= max_samples_per_shard
            )
            if hit_bytes or hit_count:
                sid = _make_deterministic_shard_id(
                    release_id, str(split_name), shard_idx, current_ids
                )
                specs.append(
                    PlannedShardSpec(
                        shard_id=sid,
                        release_split=str(split_name),
                        storage_key=f"releases/{release_id}/shards/{split_name}/{sid}.tar",
                        receipt_storage_key=f"releases/{release_id}/shards/{split_name}/{sid}.receipt.json",
                        image_ids=tuple(current_ids),
                        source_shards=tuple(sorted(current_src_shards)),
                    )
                )
                shard_idx += 1
                current_ids = []
                current_src_shards = set()
                current_est_bytes = 0

            current_ids.append(img_id)
            if pd.notna(row.get("source_shard")):
                current_src_shards.add(str(row["source_shard"]))
            current_est_bytes += est_sz

        if current_ids:
            sid = _make_deterministic_shard_id(release_id, str(split_name), shard_idx, current_ids)
            specs.append(
                PlannedShardSpec(
                    shard_id=sid,
                    release_split=str(split_name),
                    storage_key=f"releases/{release_id}/shards/{split_name}/{sid}.tar",
                    receipt_storage_key=f"releases/{release_id}/shards/{split_name}/{sid}.receipt.json",
                    image_ids=tuple(current_ids),
                    source_shards=tuple(sorted(current_src_shards)),
                )
            )

    return specs


def _row_to_source_record(row: pd.Series) -> SourceRecord:
    src_path = str(row["source_path"])
    orig_path_val = row.get("original_image_path")
    orig_path = (
        str(orig_path_val) if pd.notna(orig_path_val) and str(orig_path_val).strip() else None
    )
    ext_basis = orig_path or src_path.split("::")[-1]
    ext = Path(ext_basis).suffix.lower()

    raw_sz = row.get("file_size_bytes")
    raw_row_idx = row.get("source_row_index")
    raw_sha = row.get("sha256")
    expected_sha = str(raw_sha).strip() if pd.notna(raw_sha) and str(raw_sha).strip() else None

    return SourceRecord(
        source_id=str(row["source_id"]),
        source_revision=str(row["source_revision"]),
        source_path=src_path,
        source_split=str(row["source_split"]),
        label=int(row["label"]),
        generator=str(row["generator"]),
        source_generator_group=str(row["source_generator_group"]),
        source_shard=str(row["source_shard"]) if pd.notna(row.get("source_shard")) else None,
        imagenet_class=(
            str(row["imagenet_class"]) if pd.notna(row.get("imagenet_class")) else None
        ),
        file_size_bytes=int(raw_sz) if pd.notna(raw_sz) else None,
        file_extension=ext,
        image_id=str(row["image_id"]),
        source_row_index=int(raw_row_idx) if pd.notna(raw_row_idx) else None,
        original_image_path=orig_path,
        expected_sha256=expected_sha,
    )


def _fetch_with_retries(
    adapter: SourceAdapter,
    rec: SourceRecord,
    max_retries: int,
    base_backoff: float = 0.25,
) -> tuple[SourceImagePayload | None, str | None]:
    last_err = ""
    for attempt in range(max(1, max_retries)):
        try:
            return adapter.fetch_image_bytes(rec), None
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < max_retries and base_backoff > 0:
                time.sleep(base_backoff * (2**attempt))
    return None, last_err


def _try_resume_committed_shard(
    spec: PlannedShardSpec,
    storage: StorageBackend,
    local_staging_dir: Path,
) -> ShardReceipt | None:
    """Verify whether a shard is already committed & intact in storage (never trusting filename alone)."""
    if not storage.exists(spec.receipt_storage_key) or not storage.exists(spec.storage_key):
        return None

    local_receipt_path = local_staging_dir / f"{spec.shard_id}.receipt.json"
    local_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        storage.download(spec.receipt_storage_key, local_receipt_path)
        receipt = ShardReceipt.from_dict(json.loads(local_receipt_path.read_text(encoding="utf-8")))
    except Exception:
        return None

    if receipt.status != "COMMITTED":
        return None
    if receipt.shard_id != spec.shard_id or receipt.storage_key != spec.storage_key:
        return None

    receipt_ids = tuple(m["image_id"] for m in receipt.members)
    if set(receipt_ids) != set(spec.image_ids) or len(receipt_ids) != len(spec.image_ids):
        return None

    if not storage.verify_integrity(
        spec.storage_key,
        expected_sha256=receipt.sha256,
        expected_size=receipt.byte_size,
    ):
        return None

    return receipt


def _append_exclusion_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")


def materialize_release(
    selection_df: pd.DataFrame,
    adapter: SourceAdapter,
    storage: StorageBackend,
    config: ReleaseConfig,
    parent_receipts: list[ShardReceipt] | None = None,
    parent_manifest_df: pd.DataFrame | None = None,
    resume: bool = True,
    strict_frozen_mode: bool | None = None,
) -> MaterializationResult:
    """Materialize selected images into immutable verified TAR shards with resume & parent reuse."""
    t0 = time.perf_counter()
    strict_mode = (
        bool(strict_frozen_mode) if strict_frozen_mode is not None else config.strict_frozen_mode
    )
    work_dir = Path(config.working_dir) / config.release_id
    staging_root = work_dir / "staging_shards"
    staging_root.mkdir(parents=True, exist_ok=True)
    exclusions_path = work_dir / "exclusions.jsonl"
    if exclusions_path.exists():
        exclusions_path.unlink()

    selected_id_set = set(selection_df["image_id"].astype(str).tolist())
    all_receipts: list[ShardReceipt] = []
    reused_image_ids: set[str] = set()
    committed_image_ids: set[str] = set()
    reused_parent_rows: list[pd.DataFrame] = []
    reused_parent_shards = 0

    # 1. Reuse verified parent shards whose members are 100% present in this child release
    if parent_receipts and parent_manifest_df is not None:
        parent_by_id = parent_manifest_df.set_index("image_id", drop=False)
        for p_rec in parent_receipts:
            member_ids = [m["image_id"] for m in p_rec.members]
            if not member_ids or not set(member_ids).issubset(selected_id_set):
                continue
            if not storage.verify_integrity(
                p_rec.storage_key,
                expected_sha256=p_rec.sha256,
                expected_size=p_rec.byte_size,
            ):
                raise RuntimeError(
                    f"Parent shard '{p_rec.shard_id}' at '{p_rec.storage_key}' failed integrity check."
                )
            reused_receipt = ShardReceipt(
                shard_id=p_rec.shard_id,
                release_id=config.release_id,
                origin_release_id=p_rec.origin_release_id,
                release_split=p_rec.release_split,
                storage_key=p_rec.storage_key,
                status="COMMITTED",
                image_count=p_rec.image_count,
                byte_size=p_rec.byte_size,
                sha256=p_rec.sha256,
                created_at_utc=p_rec.created_at_utc,
                members=p_rec.members,
            )
            all_receipts.append(reused_receipt)
            reused_image_ids.update(member_ids)
            committed_image_ids.update(member_ids)
            reused_parent_rows.append(parent_by_id.loc[member_ids].copy())
            reused_parent_shards += 1

    # 2. Plan deterministic shards for newly added (delta) images only
    delta_df = selection_df[~selection_df["image_id"].astype(str).isin(reused_image_ids)].copy()
    shard_specs = plan_deterministic_shards(
        delta_df=delta_df,
        release_id=config.release_id,
        shard_target_bytes=config.shard_target_bytes,
        max_samples_per_shard=config.max_samples_per_shard,
    )

    # Track remaining pending references to each upstream source_shard so Arrow shards
    # can be proactively released from local disk as soon as they are no longer needed.
    remaining_src_shard_uses: Counter[str] = Counter()
    for sp in shard_specs:
        for src_sh in sp.source_shards:
            remaining_src_shard_uses[src_sh] += 1

    caps = adapter.capabilities()
    delta_indexed = delta_df.set_index("image_id", drop=False)
    resumed_existing_shards = 0
    newly_materialized_shards = 0
    excluded_records: list[dict[str, Any]] = []
    materialized_rows: list[dict[str, Any]] = []

    bytes_written_total = 0
    images_decoded_total = 0

    for spec in shard_specs:
        split_stage_dir = staging_root / spec.release_split
        split_stage_dir.mkdir(parents=True, exist_ok=True)

        # Resume check: skip shard only if remote receipt + shard SHA-256 verify cleanly
        if resume:
            resumed_receipt = _try_resume_committed_shard(spec, storage, split_stage_dir)
            if resumed_receipt is not None:
                overlap = committed_image_ids & set(spec.image_ids)
                if overlap:
                    raise RuntimeError(
                        f"Duplicate image commit detected during resume for shard '{spec.shard_id}': "
                        f"{sorted(overlap)[:5]}"
                    )
                committed_image_ids.update(spec.image_ids)
                all_receipts.append(resumed_receipt)
                resumed_existing_shards += 1
                for m in resumed_receipt.members:
                    base_row = delta_indexed.loc[m["image_id"]].to_dict()
                    base_row.update(
                        {
                            "width": m.get("width"),
                            "height": m.get("height"),
                            "image_format": m.get("image_format"),
                            "file_size_bytes": m["byte_size"],
                            "sha256": m["sha256"],
                            "decode_status": "OK",
                            "shard_id": resumed_receipt.shard_id,
                            "shard_storage_key": resumed_receipt.storage_key,
                            "shard_member_name": m["member_name"],
                            "shard_offset_data": m["offset_data"],
                            "origin_release_id": resumed_receipt.origin_release_id,
                        }
                    )
                    materialized_rows.append(base_row)
                for src_sh in spec.source_shards:
                    remaining_src_shard_uses[src_sh] -= 1
                    if remaining_src_shard_uses[src_sh] <= 0:
                        adapter.release_source_shard(src_sh)
                continue

        # Guard against duplicate commit across shards before opening writer
        overlap = committed_image_ids & set(spec.image_ids)
        if overlap:
            raise RuntimeError(
                f"Duplicate image_id scheduled across output shards in '{spec.shard_id}': "
                f"{sorted(overlap)[:5]}"
            )

        # Remove stale local incomplete staging files before fresh shard write
        for stale in (
            split_stage_dir / f"{spec.shard_id}.tar",
            split_stage_dir / f"{spec.shard_id}.tar.tmp",
            split_stage_dir / f"{spec.shard_id}.receipt.json",
        ):
            stale.unlink(missing_ok=True)

        shard_rows_df = delta_indexed.loc[list(spec.image_ids)]
        source_records = [_row_to_source_record(row) for _, row in shard_rows_df.iterrows()]
        rec_by_path = {
            r.source_path: (str(img_id), shard_rows_df.loc[img_id])
            for img_id, r in zip(spec.image_ids, source_records, strict=True)
        }

        writer = ImmutableTarShardWriter(
            shard_dir=split_stage_dir,
            shard_id=spec.shard_id,
            release_id=config.release_id,
            release_split=spec.release_split,
            storage_key=spec.storage_key,
        )

        shard_valid_rows: list[dict[str, Any]] = []
        shard_exclusions: list[dict[str, Any]] = []

        def _process_payload(
            img_id: str,
            row_series: pd.Series,
            payload: SourceImagePayload | None,
            err_msg: str | None,
        ) -> tuple[
            str,
            pd.Series,
            SourceImagePayload | None,
            ByteValidationResult | None,
            MaterializationFailureCategory | None,
            str | None,
        ]:
            if err_msg is not None or payload is None:
                msg = err_msg or "EMPTY_PAYLOAD"
                return img_id, row_series, None, None, _classify_error_message(msg), msg

            # Validate image decodability without enforcing filename extension prematurely
            # (use actual detected image format for extension consistency while preserving raw bytes!)
            vres = validate_image_bytes(
                raw_bytes=payload.raw_bytes,
                file_extension="",
                supported_extensions=set(config.source.supported_extensions),
                compute_pixel_hash=config.compute_pixel_hash,
                compute_perceptual_hash=config.compute_perceptual_hash,
            )
            if not vres.is_valid:
                return (
                    img_id,
                    row_series,
                    None,
                    vres,
                    _classify_error_message(vres.decode_status),
                    vres.decode_status,
                )

            # Strictly verify frozen SHA-256 hash when present in selection (NEVER overwrite mismatch!)
            frozen_sha_val = row_series.get("sha256")
            if pd.notna(frozen_sha_val) and str(frozen_sha_val).strip():
                expected_sha = str(frozen_sha_val).strip()
                if vres.sha256 != expected_sha:
                    msg = (
                        f"SHA256_MISMATCH: Fetched bytes for image_id='{img_id}' have "
                        f"sha256={vres.sha256}, expected frozen sha256={expected_sha}."
                    )
                    return (
                        img_id,
                        row_series,
                        None,
                        vres,
                        MaterializationFailureCategory.SHA256_MISMATCH,
                        msg,
                    )

            return img_id, row_series, payload, vres, None, None

        def _handle_failure(
            img_id: str,
            row_series: pd.Series,
            category: MaterializationFailureCategory,
            reason: str,
            _spec: PlannedShardSpec = spec,
            _exclusions: list[dict[str, Any]] = shard_exclusions,
        ) -> None:
            item = {
                "image_id": img_id,
                "release_split": _spec.release_split,
                "source_path": str(row_series["source_path"]),
                "source_shard": (
                    str(row_series["source_shard"])
                    if pd.notna(row_series.get("source_shard"))
                    else None
                ),
                "source_row_index": (
                    int(row_series["source_row_index"])
                    if pd.notna(row_series.get("source_row_index"))
                    else None
                ),
                "failure_category": category.value,
                "reason": reason,
            }
            _exclusions.append(item)
            _append_exclusion_jsonl(exclusions_path, item)
            if strict_mode:
                raise FrozenMaterializationError(
                    f"Strict frozen materialization failed for image_id='{img_id}' "
                    f"[{category.value}]: {reason}",
                    image_id=img_id,
                    category=category,
                    details=item,
                )

        def _record_valid_image(
            img_id: str,
            row_series: pd.Series,
            payload: SourceImagePayload,
            vres: ByteValidationResult,
            _spec: PlannedShardSpec = spec,
            _writer: ImmutableTarShardWriter = writer,
            _valid_rows: list[dict[str, Any]] = shard_valid_rows,
        ) -> None:
            nonlocal images_decoded_total
            out_ext = _resolve_output_extension(payload.file_extension, vres.image_format)
            member = _writer.add_image(
                image_id=img_id,
                raw_bytes=payload.raw_bytes,
                file_extension=out_ext,
                sha256=vres.sha256,
                label=int(row_series["label"]),
                generator=str(row_series["generator"]),
                source_generator_group=str(row_series["source_generator_group"]),
                source_split=str(row_series["source_split"]),
                width=vres.width,
                height=vres.height,
                image_format=vres.image_format,
            )
            images_decoded_total += 1
            row_dict = row_series.to_dict()
            row_dict.update(
                {
                    "width": vres.width,
                    "height": vres.height,
                    "image_format": vres.image_format,
                    "file_size_bytes": vres.file_size_bytes,
                    "sha256": vres.sha256,
                    "pixel_hash": vres.pixel_hash,
                    "perceptual_hash": vres.perceptual_hash,
                    "decode_status": "OK",
                    "shard_id": _spec.shard_id,
                    "shard_storage_key": _spec.storage_key,
                    "shard_member_name": member.member_name,
                    "shard_offset_data": member.offset_data,
                    "origin_release_id": config.release_id,
                }
            )
            _valid_rows.append(row_dict)

        try:
            if caps.random_access and config.materialization_workers > 1:
                batch_sz = config.materialization_batch_size
                for start in range(0, len(source_records), batch_sz):
                    batch_recs = source_records[start : start + batch_sz]
                    batch_ids = spec.image_ids[start : start + batch_sz]

                    def _fetch_and_validate(
                        item: tuple[str, SourceRecord],
                        rows_df: pd.DataFrame = shard_rows_df,
                    ) -> tuple[
                        str,
                        pd.Series,
                        SourceImagePayload | None,
                        ByteValidationResult | None,
                        MaterializationFailureCategory | None,
                        str | None,
                    ]:
                        iid, srec = item
                        payload, err = _fetch_with_retries(adapter, srec, config.source_max_retries)
                        return _process_payload(iid, rows_df.loc[iid], payload, err)

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=config.materialization_workers
                    ) as pool:
                        batch_results = list(
                            pool.map(
                                _fetch_and_validate,
                                list(zip(batch_ids, batch_recs, strict=True)),
                            )
                        )

                    for iid, row_s, payload, vres, cat, err in batch_results:
                        if err is not None or payload is None or vres is None:
                            _handle_failure(
                                iid,
                                row_s,
                                cat or MaterializationFailureCategory.SOURCE_FETCH_FAILURE,
                                err or "UNKNOWN_ERROR",
                            )
                            continue
                        _record_valid_image(iid, row_s, payload, vres)
            else:
                # Shard-aware sequential streaming path (Arrow shards, TAR/ZIP archives, HF streaming)
                for srec, payload, err in adapter.iter_selected_images(source_records):
                    iid, row_s = rec_by_path[srec.source_path]
                    _, _, valid_payload, vres, cat, val_err = _process_payload(
                        iid, row_s, payload, err
                    )
                    if val_err is not None or valid_payload is None or vres is None:
                        _handle_failure(
                            iid,
                            row_s,
                            cat or MaterializationFailureCategory.SOURCE_FETCH_FAILURE,
                            val_err or "UNKNOWN_ERROR",
                        )
                        continue
                    _record_valid_image(iid, row_s, valid_payload, vres)

            excluded_records.extend(shard_exclusions)

            if not shard_valid_rows:
                writer.abort()
                continue

            tar_path, receipt_path, receipt = writer.finalize()
            offset_by_id = {m["image_id"]: int(m["offset_data"]) for m in receipt.members}
            for row_dict in shard_valid_rows:
                row_dict["shard_offset_data"] = offset_by_id[str(row_dict["image_id"])]

            ok, verify_msg = verify_tar_shard_file(
                tar_path,
                expected_sha256=receipt.sha256,
                expected_byte_size=receipt.byte_size,
                expected_image_count=receipt.image_count,
            )
            if not ok:
                raise RuntimeError(f"Post-finalize verification failed: {verify_msg}")

            # Clean up any partial/corrupt uncommitted remote artifacts from a prior interrupted run.
            try:
                if storage.exists(spec.storage_key):
                    storage.delete_uncommitted(spec.storage_key)
                if storage.exists(spec.receipt_storage_key):
                    storage.delete_uncommitted(spec.receipt_storage_key)

                # Upload immutable shard first, then receipt (receipt signals committed shard)
                storage.upload(
                    local_path=tar_path,
                    remote_key=spec.storage_key,
                    expected_sha256=receipt.sha256,
                    overwrite=False,
                )
                storage.upload(
                    local_path=receipt_path,
                    remote_key=spec.receipt_storage_key,
                    overwrite=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{MaterializationFailureCategory.STORAGE_FAILURE.value}: "
                    f"Failed uploading shard '{spec.shard_id}': {type(exc).__name__}: {exc}"
                ) from exc

            bytes_written_total += receipt.byte_size
            newly_materialized_shards += 1
            all_receipts.append(receipt)
            committed_image_ids.update(r["image_id"] for r in shard_valid_rows)
            materialized_rows.extend(shard_valid_rows)

            # Free local staging tar once verified & uploaded to keep bounded disk usage
            tar_path.unlink(missing_ok=True)
        except Exception:
            writer.abort()
            raise
        finally:
            for src_sh in spec.source_shards:
                remaining_src_shard_uses[src_sh] -= 1
                if remaining_src_shard_uses[src_sh] <= 0:
                    adapter.release_source_shard(src_sh)

    frames: list[pd.DataFrame] = []
    if reused_parent_rows:
        frames.extend(reused_parent_rows)
    if materialized_rows:
        frames.append(pd.DataFrame(materialized_rows))

    if frames:
        combined_df = pd.concat(frames, ignore_index=True)
        combined_df = combined_df.sort_values(
            by=["release_split", "image_id"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        combined_df = selection_df.iloc[0:0].copy()

    # Re-run exact SHA-256 & perceptual near-duplicate candidate audit on materialized image bytes.
    # Exact duplicates and perceptual similarity candidates are strictly separated in DuplicateAuditReport.
    updated_df, dup_report = assign_leakage_groups(
        combined_df,
        audit_scope="materialized_release_images_only",
        include_pixel_hash=config.compute_pixel_hash,
        include_perceptual_in_leakage_groups=config.protocol.include_perceptual_in_leakage_groups,
        perceptual_hamming_threshold=config.protocol.perceptual_hamming_threshold,
    )

    # Identify cross-split perceptual near-duplicate candidates explicitly (informational/audit gate)
    cross_split_near_dups: list[dict[str, Any]] = []
    if dup_report.sample_near_duplicate_candidates and not updated_df.empty:
        split_by_id = dict(
            zip(
                updated_df["image_id"].astype(str).tolist(),
                updated_df["release_split"].astype(str).tolist(),
                strict=True,
            )
        )
        for cand in dup_report.sample_near_duplicate_candidates:
            sa = split_by_id.get(cand["image_id_a"])
            sb = split_by_id.get(cand["image_id_b"])
            if sa and sb and sa != sb:
                cross_split_near_dups.append({**cand, "split_a": sa, "split_b": sb})

    duration = time.perf_counter() - t0
    metrics = {
        "duration_seconds": round(duration, 4),
        "images_decoded_and_hashed": images_decoded_total,
        "images_per_second": round(images_decoded_total / max(duration, 1e-6), 2),
        "bytes_written_to_new_shards": bytes_written_total,
        "megabytes_per_second": round(
            (bytes_written_total / (1024 * 1024)) / max(duration, 1e-6), 2
        ),
        "reused_parent_shards": reused_parent_shards,
        "resumed_existing_shards": resumed_existing_shards,
        "newly_materialized_shards": newly_materialized_shards,
        "excluded_count": len(excluded_records),
        "strict_frozen_mode": strict_mode,
        "cross_split_perceptual_candidates": len(cross_split_near_dups),
        "cross_split_perceptual_candidates_sample": cross_split_near_dups[:20],
        "adapter_runtime_stats": adapter.runtime_stats(),
    }

    return MaterializationResult(
        release_id=config.release_id,
        manifest_df=updated_df,
        receipts=all_receipts,
        reused_parent_shards=reused_parent_shards,
        resumed_existing_shards=resumed_existing_shards,
        newly_materialized_shards=newly_materialized_shards,
        excluded_records=excluded_records,
        duplicate_report=dup_report,
        duration_seconds=duration,
        metrics=metrics,
    )
