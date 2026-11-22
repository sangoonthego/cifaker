"""Bounded-memory streaming dataset inventory builder writing Parquet + summary JSON.

Streams `SourceRecord` objects from any `SourceAdapter` in fixed-size batches,
assigns deterministic canonical `image_id`s, validates each batch against the
canonical schema, and writes incrementally to a Parquet file via `ParquetWriter`
so memory usage remains O(batch_size) even for multi-million-image upstream datasets.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from prooflens.data.registry.schema import (
    CANONICAL_ARROW_SCHEMA,
    CanonicalImageRecord,
    dataframe_to_canonical_arrow,
    make_canonical_image_id,
    validate_canonical_dataframe,
)
from prooflens.data.sources.base import SourceAdapter


@dataclass(frozen=True)
class InventoryResult:
    """Outcome summary returned by `build_source_inventory`."""

    parquet_path: Path
    summary_path: Path
    total_records: int
    eligible_records: int
    excluded_records: int
    duration_seconds: float
    summary: dict[str, Any]


def build_source_inventory(
    adapter: SourceAdapter,
    output_parquet: Path | str,
    summary_json: Path | str | None = None,
    batch_size: int = 5000,
    validate_image_bytes: bool = False,
    compute_pixel_hash: bool = False,
    compute_perceptual_hash: bool = False,
) -> InventoryResult:
    """Enumerate an upstream source in bounded-memory batches and persist canonical Parquet.

    Args:
        adapter: Configured `SourceAdapter`.
        output_parquet: Destination `.parquet` path.
        summary_json: Optional JSON summary destination (defaults to `<parquet>.summary.json`).
        batch_size: Maximum rows buffered in memory before flushing a Parquet row group.
        validate_image_bytes: If True, also fetch and validate image bytes (decoding + SHA-256)
            during inventory. Keep False for fast metadata-only enumeration before sampling.
        compute_pixel_hash: When `validate_image_bytes=True`, also compute RGB pixel SHA-256.
        compute_perceptual_hash: When `validate_image_bytes=True`, also compute 64-bit dHash.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(summary_json) if summary_json is not None else out_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    caps = adapter.capabilities()
    t0 = time.perf_counter()

    seen_ids: set[str] = set()
    total_records = 0
    eligible_records = 0
    excluded_records = 0
    total_known_bytes = 0
    records_with_size = 0

    split_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    generator_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    group_split_label_counts: Counter[str] = Counter()
    class_present_count = 0

    # Lazy import validator only if byte validation requested
    byte_validator = None
    if validate_image_bytes:
        from prooflens.data.processing.validator import validate_image_bytes as _val_bytes

        byte_validator = _val_bytes

    staging_parquet = out_path.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None

    def _flush_batch(rows: list[dict[str, Any]]) -> None:
        nonlocal writer
        if not rows:
            return
        batch_df = pd.DataFrame(rows)
        validated_df = validate_canonical_dataframe(batch_df)
        table = dataframe_to_canonical_arrow(validated_df)
        if writer is None:
            writer = pq.ParquetWriter(staging_parquet, CANONICAL_ARROW_SCHEMA, compression="snappy")
        writer.write_table(table)

    batch_rows: list[dict[str, Any]] = []
    try:
        for src_rec in adapter.iter_records():
            img_id = make_canonical_image_id(
                src_rec.source_id, src_rec.source_revision, src_rec.source_path
            )
            if img_id in seen_ids:
                raise ValueError(
                    f"Duplicate canonical image_id '{img_id}' generated for source_path "
                    f"'{src_rec.source_path}' during inventory."
                )
            seen_ids.add(img_id)

            width = None
            height = None
            image_format = src_rec.original_format
            file_size = src_rec.file_size_bytes
            sha256_hex = None
            pixel_hash = None
            perceptual_hash = None
            decode_status = None
            exclusion_reason = None
            eligibility = "ELIGIBLE"

            if validate_image_bytes and byte_validator is not None:
                try:
                    payload = adapter.fetch_image_bytes(src_rec)
                    vres = byte_validator(
                        payload.raw_bytes,
                        file_extension=payload.file_extension,
                        compute_pixel_hash=compute_pixel_hash,
                        compute_perceptual_hash=compute_perceptual_hash,
                    )
                    width = vres.width
                    height = vres.height
                    image_format = vres.image_format
                    file_size = vres.file_size_bytes
                    sha256_hex = vres.sha256
                    pixel_hash = vres.pixel_hash
                    perceptual_hash = vres.perceptual_hash
                    decode_status = vres.decode_status
                    if not vres.is_valid:
                        eligibility = "EXCLUDED"
                        exclusion_reason = vres.decode_status
                except Exception as exc:
                    eligibility = "EXCLUDED"
                    decode_status = f"FETCH_ERROR:{type(exc).__name__}"
                    exclusion_reason = str(exc)

            # Leakage group defaults to exact sha256 when available, else unique canonical id
            leakage_group = f"lg_sha_{sha256_hex[:16]}" if sha256_hex else f"lg_{img_id}"

            canonical = CanonicalImageRecord(
                image_id=img_id,
                source_id=src_rec.source_id,
                source_revision=src_rec.source_revision,
                source_path=src_rec.source_path,
                source_split=src_rec.source_split,
                label=src_rec.label,
                generator=src_rec.generator,
                source_generator_group=src_rec.source_generator_group,
                source_shard=src_rec.source_shard,
                imagenet_class=src_rec.imagenet_class,
                leakage_group_id=leakage_group,
                eligibility_status=eligibility,
                width=width,
                height=height,
                image_format=image_format,
                file_size_bytes=file_size,
                sha256=sha256_hex,
                pixel_hash=pixel_hash,
                perceptual_hash=perceptual_hash,
                decode_status=decode_status,
                exclusion_reason=exclusion_reason,
            )
            batch_rows.append(canonical.to_dict())

            total_records += 1
            if eligibility == "ELIGIBLE":
                eligible_records += 1
            else:
                excluded_records += 1

            if file_size is not None:
                total_known_bytes += int(file_size)
                records_with_size += 1
            if src_rec.imagenet_class is not None:
                class_present_count += 1

            split_counts[src_rec.source_split] += 1
            label_counts[src_rec.label] += 1
            generator_counts[src_rec.generator] += 1
            group_counts[src_rec.source_generator_group] += 1
            group_split_label_counts[
                f"{src_rec.source_generator_group}|{src_rec.source_split}|{src_rec.label}"
            ] += 1

            if len(batch_rows) >= batch_size:
                _flush_batch(batch_rows)
                batch_rows.clear()

        _flush_batch(batch_rows)
        batch_rows.clear()

        if writer is None:
            # Empty source -> still write valid empty Parquet table with canonical schema
            empty_table = dataframe_to_canonical_arrow(
                pd.DataFrame(columns=list(CANONICAL_ARROW_SCHEMA.names))
            )
            pq.write_table(empty_table, staging_parquet, compression="snappy")
        else:
            writer.close()
            writer = None

        staging_parquet.replace(out_path)
    finally:
        if writer is not None:
            writer.close()
        if staging_parquet.exists():
            staging_parquet.unlink(missing_ok=True)

    duration = time.perf_counter() - t0
    avg_bytes = (total_known_bytes / records_with_size) if records_with_size > 0 else None

    summary: dict[str, Any] = {
        "source_id": adapter.source_id,
        "source_revision": adapter.source_revision,
        "inventory_parquet": str(out_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": round(duration, 4),
        "records_per_second": round(total_records / max(duration, 1e-6), 2),
        "bytes_validated_during_inventory": validate_image_bytes,
        "source_capabilities": caps.to_dict(),
        "total_records": total_records,
        "eligible_records": eligible_records,
        "excluded_records": excluded_records,
        "records_with_known_file_size": records_with_size,
        "total_known_bytes": total_known_bytes if records_with_size > 0 else None,
        "mean_known_file_size_bytes": round(avg_bytes, 2) if avg_bytes is not None else None,
        "imagenet_class_coverage": round(class_present_count / max(total_records, 1), 4),
        "source_split_counts": dict(sorted(split_counts.items())),
        "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
        "generator_counts": dict(sorted(generator_counts.items())),
        "source_generator_group_counts": dict(sorted(group_counts.items())),
        "strata_counts": dict(sorted(group_split_label_counts.items())),
        "duplicate_audit_scope": (
            "full_inventory_sha256" if validate_image_bytes else "metadata_only_not_hashed"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return InventoryResult(
        parquet_path=out_path,
        summary_path=summary_path,
        total_records=total_records,
        eligible_records=eligible_records,
        excluded_records=excluded_records,
        duration_seconds=duration,
        summary=summary,
    )
