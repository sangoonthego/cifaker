"""Read-only local and remote storage capacity preflight for dataset materialization.

Accounts explicitly for:
- Estimated original selected image bytes
- TAR header & 512-byte block padding overhead (~1024 B/image)
- Manifest, receipts, and audit JSON metadata overhead
- Temporary in-flight remote upload retry overhead
- Local runtime staging disk requirement (temporary Arrow source shards + output TAR shard)
- Configurable safety margin

When remote storage quota cannot be verified (e.g., Google Drive shared/workspace drive
or unauthenticated API), reports `remote_status = "UNKNOWN"` and `requires_explicit_approval = True`
instead of assuming unlimited capacity.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prooflens.data.config import ReleaseConfig
from prooflens.data.storage.base import StorageBackend

_TAR_OVERHEAD_BYTES_PER_IMAGE = 1024
_DEFAULT_ARROW_SHARD_EST_BYTES = 600 * 1024 * 1024  # Conservative ~600 MiB per HF Arrow shard


@dataclass(frozen=True)
class CapacityPreflightReport:
    """Structured storage capacity preflight evaluation."""

    release_id: str
    selected_image_count: int
    unique_source_shards: int
    estimated_original_image_bytes: int | None
    original_image_bytes_known_ratio: float
    tar_overhead_bytes: int
    metadata_overhead_bytes: int
    remote_retry_overhead_bytes: int
    remote_required_bytes_with_margin: int | None
    local_temp_arrow_bytes: int
    local_temp_output_shard_bytes: int
    local_required_bytes_with_margin: int
    safety_margin_ratio: float
    remote_capacity: dict[str, Any]
    local_capacity: dict[str, Any]
    remote_status: str  # "PASS", "FAIL", "UNKNOWN"
    local_status: str  # "PASS", "FAIL"
    overall_status: str  # "PASS", "FAIL", "REQUIRES_EXPLICIT_APPROVAL"
    requires_explicit_approval: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_capacity_preflight(
    selection_df: pd.DataFrame,
    config: ReleaseConfig,
    storage: StorageBackend,
    safety_margin_ratio: float = 0.15,
    fallback_mean_image_bytes: int = 256 * 1024,
) -> CapacityPreflightReport:
    """Run a read-only capacity preflight check across local staging disk and target storage."""
    if safety_margin_ratio < 0.0:
        raise ValueError(f"safety_margin_ratio must be >= 0, got {safety_margin_ratio}.")

    n_images = int(len(selection_df))
    n_shards = (
        int(selection_df["source_shard"].dropna().nunique())
        if "source_shard" in selection_df.columns
        else 0
    )
    warnings: list[str] = []

    known_sizes = (
        selection_df["file_size_bytes"].dropna()
        if "file_size_bytes" in selection_df.columns
        else pd.Series(dtype=float)
    )
    known_ratio = float(len(known_sizes) / max(n_images, 1)) if n_images > 0 else 1.0

    if n_images == 0:
        est_image_bytes: int | None = 0
    elif len(known_sizes) == n_images:
        est_image_bytes = int(known_sizes.sum())
    elif len(known_sizes) > 0:
        mean_sz = float(known_sizes.mean())
        est_image_bytes = int(mean_sz * n_images)
        warnings.append(
            f"Only {len(known_sizes)}/{n_images} ({known_ratio:.1%}) rows have file_size_bytes; "
            "original image storage is extrapolated from observed mean."
        )
    else:
        est_image_bytes = int(fallback_mean_image_bytes * n_images)
        warnings.append(
            "No file_size_bytes found in selection; using fallback heuristic "
            f"({fallback_mean_image_bytes} B/image). Treat storage estimate as approximate."
        )

    tar_overhead = n_images * _TAR_OVERHEAD_BYTES_PER_IMAGE
    metadata_overhead = int(50 * 1024 * 1024 + 0.01 * (est_image_bytes or 0))
    # Cap in-flight retry overhead at min(shard_target_bytes, total estimated payload)
    effective_shard_bytes = min(
        config.shard_target_bytes,
        max(1024 * 1024, (est_image_bytes or 0) + tar_overhead),
    )
    remote_retry_overhead = effective_shard_bytes

    remote_base = (est_image_bytes or 0) + tar_overhead + metadata_overhead + remote_retry_overhead
    remote_required = int(remote_base * (1.0 + safety_margin_ratio))

    # Local staging disk requirement (temporary Arrow source cache + 1 staging output TAR + metadata)
    if config.source.source_type == "huggingface_arrow":
        if config.source.arrow_max_cache_bytes is not None:
            local_temp_arrow = int(config.source.arrow_max_cache_bytes)
        else:
            local_temp_arrow = int(
                config.source.arrow_max_cached_shards * _DEFAULT_ARROW_SHARD_EST_BYTES
            )
    else:
        local_temp_arrow = 0

    local_temp_out = effective_shard_bytes
    local_base = local_temp_arrow + local_temp_out + metadata_overhead
    local_required = int(local_base * (1.0 + safety_margin_ratio))

    # Query remote storage capacity (read-only)
    remote_cap = storage.get_capacity_info()
    if remote_cap.status == "KNOWN" and remote_cap.free_bytes is not None:
        if remote_cap.free_bytes >= remote_required:
            remote_status = "PASS"
        else:
            remote_status = "FAIL"
            warnings.append(
                f"Insufficient remote storage free space on {remote_cap.backend_name}: "
                f"free={remote_cap.free_bytes:,} B < required={remote_required:,} B."
            )
    else:
        remote_status = "UNKNOWN"
        warnings.append(
            f"Remote storage capacity on {remote_cap.backend_name} is UNKNOWN ({remote_cap.notes}). "
            "Explicit operator confirmation is required before full materialization."
        )

    # Query local runtime staging disk capacity
    work_path = Path(config.working_dir).resolve()
    probe_path = work_path
    while not probe_path.exists() and probe_path.parent != probe_path:
        probe_path = probe_path.parent
    local_usage = shutil.disk_usage(probe_path)
    local_cap_dict = {
        "probe_path": str(probe_path),
        "total_bytes": int(local_usage.total),
        "used_bytes": int(local_usage.used),
        "free_bytes": int(local_usage.free),
    }
    if int(local_usage.free) >= local_required:
        local_status = "PASS"
    else:
        local_status = "FAIL"
        warnings.append(
            f"Insufficient local staging disk space at '{probe_path}': "
            f"free={int(local_usage.free):,} B < required={local_required:,} B."
        )

    if local_status == "FAIL" or remote_status == "FAIL":
        overall_status = "FAIL"
        requires_approval = False
    elif remote_status == "UNKNOWN":
        overall_status = "REQUIRES_EXPLICIT_APPROVAL"
        requires_approval = True
    else:
        overall_status = "PASS"
        requires_approval = False

    return CapacityPreflightReport(
        release_id=config.release_id,
        selected_image_count=n_images,
        unique_source_shards=n_shards,
        estimated_original_image_bytes=est_image_bytes,
        original_image_bytes_known_ratio=round(known_ratio, 4),
        tar_overhead_bytes=tar_overhead,
        metadata_overhead_bytes=metadata_overhead,
        remote_retry_overhead_bytes=remote_retry_overhead,
        remote_required_bytes_with_margin=remote_required,
        local_temp_arrow_bytes=local_temp_arrow,
        local_temp_output_shard_bytes=local_temp_out,
        local_required_bytes_with_margin=local_required,
        safety_margin_ratio=safety_margin_ratio,
        remote_capacity=remote_cap.to_dict(),
        local_capacity=local_cap_dict,
        remote_status=remote_status,
        local_status=local_status,
        overall_status=overall_status,
        requires_explicit_approval=requires_approval,
        warnings=warnings,
    )
