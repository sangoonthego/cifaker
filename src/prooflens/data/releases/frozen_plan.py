"""Frozen sampling snapshot consumer, provenance separation, and pilot selection.

Enables the production materializer to consume an externally frozen selection snapshot
(e.g., created in Colab Cells 1-4.3 and stored under Google Drive `pipeline_metadata/sampling_freeze/...`)
WITHOUT regenerating the selection, WITHOUT forging Plan IDs, and WITHOUT losing original
sampling provenance when runtime/adapter configurations change.

Explicitly separates:
- Sampling provenance (immutable):
    sampling_plan_id, sampling_config_hash, sampling_code_commit, source_id, source_revision,
    protocol_version, sampling_seed, snapshot_prefix
- Materialization provenance (runtime-specific):
    materialization_config_hash, materialization_code_commit, materialization_runtime
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from prooflens.data.config import ReleaseConfig
from prooflens.data.processing.shard_writer import compute_file_sha256
from prooflens.data.registry.schema import (
    LABEL_FAKE,
    LABEL_REAL,
    make_canonical_image_id,
    read_canonical_parquet,
)
from prooflens.data.sampling.leakage_guard import check_leakage_boundaries
from prooflens.data.sampling.planner import _compute_plan_id
from prooflens.data.sources.genimage_arrow import (
    parse_arrow_shard_path,
    parse_arrow_source_path,
)
from prooflens.data.storage.base import StorageBackend

REQUIRED_FROZEN_ARTIFACTS: tuple[str, ...] = (
    "freeze.json",
    "plan.json",
    "plan_selection.parquet",
    "resolved_config.json",
)


class FrozenSnapshotError(RuntimeError):
    """Raised when a frozen snapshot fails integrity, checksum, schema, or provenance validation."""


class FrozenPlanCompatibilityError(ValueError):
    """Raised when a materialization config violates immutable semantic fields of a frozen plan."""


@dataclass(frozen=True)
class FrozenSnapshot:
    """Validated, immutable frozen selection snapshot ready for materialization."""

    snapshot_location: str
    freeze_meta: dict[str, Any]
    plan_meta: dict[str, Any]
    resolved_sampling_config: dict[str, Any]
    selection_df: pd.DataFrame
    sampling_provenance: dict[str, Any]
    validation_report: dict[str, Any]


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of validating a materialization ReleaseConfig against a FrozenSnapshot."""

    compatible: bool
    is_pilot: bool
    sampling_provenance: dict[str, Any]
    materialization_provenance: dict[str, Any]
    checked_immutable_fields: dict[str, bool]
    authorized_runtime_differences: dict[str, dict[str, Any]]
    incompatibilities: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=3
        ).strip()
        return out or None
    except Exception:
        return None


def _extract_expected_artifact_digests(
    freeze_meta: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Normalize artifact checksum/size entries across common `freeze.json` layouts."""
    expected: dict[str, dict[str, Any]] = {}

    artifacts = freeze_meta.get("artifacts")
    if isinstance(artifacts, dict):
        for name, info in artifacts.items():
            if isinstance(info, dict):
                expected[str(name)] = {
                    "sha256": info.get("sha256"),
                    "size_bytes": (
                        info.get("size_bytes")
                        if info.get("size_bytes") is not None
                        else (
                            info.get("byte_size")
                            if info.get("byte_size") is not None
                            else info.get("size")
                        )
                    ),
                }
            elif isinstance(info, str):
                expected[str(name)] = {"sha256": info, "size_bytes": None}
    elif isinstance(artifacts, list):
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("filename") or item.get("artifact")
            if name:
                expected[str(name)] = {
                    "sha256": item.get("sha256"),
                    "size_bytes": (
                        item.get("size_bytes")
                        if item.get("size_bytes") is not None
                        else (
                            item.get("byte_size")
                            if item.get("byte_size") is not None
                            else item.get("size")
                        )
                    ),
                }

    checksums = freeze_meta.get("checksums")
    if isinstance(checksums, dict):
        for name, sha in checksums.items():
            entry = expected.setdefault(str(name), {"sha256": None, "size_bytes": None})
            if isinstance(sha, dict):
                entry["sha256"] = entry["sha256"] or sha.get("sha256")
                if entry["size_bytes"] is None:
                    entry["size_bytes"] = sha.get("size_bytes") or sha.get("byte_size")
            elif isinstance(sha, str):
                entry["sha256"] = entry["sha256"] or sha

    return expected


def _stage_snapshot_files(
    snapshot_location: str | Path,
    storage: StorageBackend | None,
    stage_dir: Path,
) -> dict[str, Path]:
    """Copy or download required (and any referenced optional) snapshot files into `stage_dir`."""
    local_files: dict[str, Path] = {}

    if storage is None:
        src_dir = Path(snapshot_location)
        if not src_dir.is_dir():
            raise FrozenSnapshotError(f"Local frozen snapshot directory not found: {src_dir}")
        for req_name in REQUIRED_FROZEN_ARTIFACTS:
            p = src_dir / req_name
            if not p.is_file():
                raise FrozenSnapshotError(
                    f"Frozen snapshot '{src_dir}' missing required artifact '{req_name}'."
                )
            local_files[req_name] = p
        # Include optional CSV artifacts if present
        for opt_name in ("quota_validation.csv", "split_summary.csv"):
            p = src_dir / opt_name
            if p.is_file():
                local_files[opt_name] = p
        return local_files

    prefix = str(snapshot_location).strip().strip("/")
    for req_name in REQUIRED_FROZEN_ARTIFACTS:
        remote_key = f"{prefix}/{req_name}"
        if not storage.exists(remote_key):
            raise FrozenSnapshotError(
                f"Remote frozen snapshot missing required artifact '{remote_key}'."
            )
        dst = stage_dir / req_name
        storage.download(remote_key, dst)
        local_files[req_name] = dst

    for opt_name in ("quota_validation.csv", "split_summary.csv"):
        remote_key = f"{prefix}/{opt_name}"
        if storage.exists(remote_key):
            dst = stage_dir / opt_name
            storage.download(remote_key, dst)
            local_files[opt_name] = dst

    return local_files


def load_and_validate_frozen_snapshot(
    snapshot_location: str | Path,
    storage: StorageBackend | None = None,
    expected_total_records: int | None = None,
    expected_source_id: str | None = None,
    expected_source_revision: str | None = None,
    expected_plan_id: str | None = None,
    verify_arrow_locators: bool = True,
) -> FrozenSnapshot:
    """Load and rigorously validate a frozen sampling snapshot from local disk or remote storage.

    Checks:
    1. Required artifacts (`freeze.json`, `plan.json`, `plan_selection.parquet`, `resolved_config.json`) exist.
    2. All artifact SHA-256 checksums and byte sizes declared in `freeze.json` match actual files.
    3. Canonical schema validation on `plan_selection.parquet` (required columns, binary labels, REAL provenance).
    4. Exact record count matches `plan.json["selected_total"]` (and `expected_total_records` when supplied).
    5. Unique `image_id`s, unique `sha256`s (when populated), and unique source locators (`source_path`
       and `(source_shard, source_row_index)` when present).
    6. Canonical `image_id` values reconcile with `make_canonical_image_id(source_id, source_revision, source_path)`.
    7. Uniform `source_id` and `source_revision` matching `resolved_config.json` / `freeze.json` / expectations.
    8. Exact split quotas (REAL/FAKE counts) and generator pool isolation (seen vs unseen).
    9. Zero cross-split duplicate or leakage-group crossings.
    10. Original `plan_id` matches `_compute_plan_id(release_id, sampling_config_hash, selection_df)`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp)
        staged = _stage_snapshot_files(snapshot_location, storage, stage_dir)

        freeze_meta = json.loads(staged["freeze.json"].read_text(encoding="utf-8"))
        plan_meta = json.loads(staged["plan.json"].read_text(encoding="utf-8"))
        resolved_cfg = json.loads(staged["resolved_config.json"].read_text(encoding="utf-8"))

        # 1. Verify declared artifact checksums & sizes from freeze.json
        expected_digests = _extract_expected_artifact_digests(freeze_meta)
        artifact_checks: dict[str, dict[str, Any]] = {}
        for art_name, spec in sorted(expected_digests.items()):
            if art_name == "freeze.json":
                continue
            art_path = staged.get(art_name)
            if art_path is None:
                # Download any extra artifact referenced in freeze.json if remote
                if storage is not None:
                    remote_key = f"{str(snapshot_location).strip().strip('/')}/{art_name}"
                    if not storage.exists(remote_key):
                        raise FrozenSnapshotError(
                            f"Artifact '{art_name}' listed in freeze.json is missing at '{remote_key}'."
                        )
                    art_path = stage_dir / art_name
                    storage.download(remote_key, art_path)
                else:
                    art_path = Path(snapshot_location) / art_name
                    if not art_path.is_file():
                        raise FrozenSnapshotError(
                            f"Artifact '{art_name}' listed in freeze.json is missing at '{art_path}'."
                        )
            actual_size = art_path.stat().st_size
            actual_sha = compute_file_sha256(art_path)
            exp_sha = spec.get("sha256")
            exp_size = spec.get("size_bytes")

            if exp_sha is not None and actual_sha != str(exp_sha):
                raise FrozenSnapshotError(
                    f"Artifact '{art_name}' SHA-256 mismatch: expected {exp_sha}, got {actual_sha}."
                )
            if exp_size is not None and actual_size != int(exp_size):
                raise FrozenSnapshotError(
                    f"Artifact '{art_name}' size mismatch: expected {exp_size} bytes, got {actual_size}."
                )
            artifact_checks[art_name] = {
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "verified": True,
            }

        # 2. Load & validate canonical selection DataFrame
        selection_df = read_canonical_parquet(staged["plan_selection.parquet"], validate=True)

    if "release_split" not in selection_df.columns:
        raise FrozenSnapshotError(
            "Frozen 'plan_selection.parquet' is missing required 'release_split' column."
        )
    if selection_df.empty:
        raise FrozenSnapshotError("Frozen 'plan_selection.parquet' contains 0 records.")

    # 3. Record count checks
    plan_selected_total = int(plan_meta.get("selected_total", len(selection_df)))
    if len(selection_df) != plan_selected_total:
        raise FrozenSnapshotError(
            f"Record count mismatch: plan.json says selected_total={plan_selected_total}, "
            f"but plan_selection.parquet has {len(selection_df)} rows."
        )
    if expected_total_records is not None and len(selection_df) != expected_total_records:
        raise FrozenSnapshotError(
            f"Expected {expected_total_records} frozen records, found {len(selection_df)}."
        )

    # 4. Unique image_id, unique SHA-256, unique source locators
    if selection_df["image_id"].duplicated().any():
        dups = selection_df.loc[selection_df["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise FrozenSnapshotError(f"Duplicate image_id in frozen selection: {dups}")

    if "sha256" in selection_df.columns and selection_df["sha256"].notna().any():
        if selection_df["sha256"].isna().any():
            raise FrozenSnapshotError(
                "Frozen selection has partial null 'sha256' values; expected complete hashes."
            )
        dup_shas = selection_df["sha256"].duplicated()
        if dup_shas.any():
            examples = selection_df.loc[dup_shas, "image_id"].head(5).tolist()
            raise FrozenSnapshotError(
                f"Duplicate SHA-256 hash detected in frozen selection (sample image_ids: {examples})."
            )

    if selection_df["source_path"].duplicated().any():
        dups = (
            selection_df.loc[selection_df["source_path"].duplicated(), "source_path"]
            .head(5)
            .tolist()
        )
        raise FrozenSnapshotError(f"Duplicate source_path locator in frozen selection: {dups}")

    has_row_idx = (
        "source_row_index" in selection_df.columns
        and selection_df["source_row_index"].notna().any()
    )
    if has_row_idx:
        if selection_df["source_row_index"].isna().any():
            raise FrozenSnapshotError(
                "Frozen selection has partial null 'source_row_index' values."
            )
        loc_dups = selection_df.duplicated(subset=["source_shard", "source_row_index"])
        if loc_dups.any():
            examples = selection_df.loc[
                loc_dups, ["image_id", "source_shard", "source_row_index"]
            ].head(5)
            raise FrozenSnapshotError(
                f"Duplicate (source_shard, source_row_index) locator in frozen selection: "
                f"{examples.to_dict(orient='records')}"
            )

    # 5. Source identity & revision consistency
    src_ids = selection_df["source_id"].astype(str).unique().tolist()
    src_revs = selection_df["source_revision"].astype(str).unique().tolist()
    if len(src_ids) != 1 or len(src_revs) != 1:
        raise FrozenSnapshotError(
            f"Frozen selection must have a single uniform (source_id, source_revision); "
            f"found source_ids={src_ids}, source_revisions={src_revs}."
        )
    actual_source_id = src_ids[0]
    actual_source_revision = src_revs[0]

    cfg_source = resolved_cfg.get("source", {})
    cfg_source_id = cfg_source.get("source_id")
    cfg_source_rev = cfg_source.get("source_revision")
    if cfg_source_id and actual_source_id != cfg_source_id:
        raise FrozenSnapshotError(
            f"Source ID mismatch: selection has '{actual_source_id}', resolved_config has '{cfg_source_id}'."
        )
    if cfg_source_rev and actual_source_revision != cfg_source_rev:
        raise FrozenSnapshotError(
            f"Source revision mismatch: selection has '{actual_source_revision}', "
            f"resolved_config has '{cfg_source_rev}'."
        )
    if expected_source_id is not None and actual_source_id != expected_source_id:
        raise FrozenSnapshotError(
            f"Expected source_id='{expected_source_id}', found '{actual_source_id}'."
        )
    if expected_source_revision is not None and actual_source_revision != expected_source_revision:
        raise FrozenSnapshotError(
            f"Expected source_revision='{expected_source_revision}', found '{actual_source_revision}'."
        )

    # 6. Verify canonical image_id derivation & Arrow locator format when applicable
    has_orig_path = (
        "original_image_path" in selection_df.columns
        and selection_df["original_image_path"].notna().any()
    )
    is_arrow_selection = has_row_idx or any(
        str(s).endswith(".arrow") for s in selection_df["source_shard"].dropna().head(5).tolist()
    )

    for row in selection_df.itertuples(index=False):
        expected_id = make_canonical_image_id(
            actual_source_id, actual_source_revision, str(row.source_path)
        )
        if str(row.image_id) != expected_id:
            raise FrozenSnapshotError(
                f"Canonical image_id mismatch for '{row.source_path}': "
                f"recorded '{row.image_id}', expected '{expected_id}'."
            )
        if verify_arrow_locators and is_arrow_selection:
            shard_str = str(row.source_shard) if row.source_shard is not None else ""
            parse_arrow_shard_path(
                shard_str,
                split_mapping=cfg_source.get("split_mapping"),
                generator_mapping=cfg_source.get("generator_mapping"),
                strict_generator_mapping=False,
            )
            row_idx_val = int(row.source_row_index) if has_row_idx else None
            orig_val = (
                str(row.original_image_path)
                if has_orig_path and pd.notna(row.original_image_path)
                else None
            )
            parse_arrow_source_path(
                str(row.source_path),
                expected_source_shard=shard_str,
                expected_row_index=row_idx_val,
                expected_original_path=orig_val,
            )

    # 7. Verify split quotas & seen/unseen generator isolation
    plan_splits = plan_meta.get("splits", {})
    cfg_proto = resolved_cfg.get("protocol", {})
    seen_gens = set(cfg_proto.get("seen_generators", []))
    unseen_gens = set(cfg_proto.get("unseen_generators", []))

    observed_split_counts: dict[str, dict[str, int]] = {}
    for sp_name, sp_spec in sorted(plan_splits.items()):
        sub = selection_df[selection_df["release_split"] == sp_name]
        act_real = int((sub["label"] == LABEL_REAL).sum())
        act_fake = int((sub["label"] == LABEL_FAKE).sum())
        req_real = int(sp_spec["requested_real"])
        req_fake = int(sp_spec["requested_fake"])
        observed_split_counts[sp_name] = {
            "real": act_real,
            "fake": act_fake,
            "total": len(sub),
        }
        if act_real != req_real or act_fake != req_fake:
            raise FrozenSnapshotError(
                f"Frozen split '{sp_name}' quota mismatch: plan expects (real={req_real}, fake={req_fake}), "
                f"selection has (real={act_real}, fake={act_fake})."
            )
        gen_pool = str(sp_spec.get("generator_pool", "")).lower()
        sub_gens = set(sub["source_generator_group"].astype(str).unique().tolist())
        if gen_pool == "seen" and seen_gens and not sub_gens.issubset(seen_gens):
            raise FrozenSnapshotError(
                f"Split '{sp_name}' contains non-seen generator group(s): {sorted(sub_gens - seen_gens)}."
            )
        if gen_pool == "unseen" and unseen_gens and not sub_gens.issubset(unseen_gens):
            raise FrozenSnapshotError(
                f"Split '{sp_name}' contains non-unseen generator group(s): {sorted(sub_gens - unseen_gens)}."
            )

        # Verify generator strata quotas if recorded in plan.json
        for stratum_info in sp_spec.get("strata", []):
            stratum_key = str(stratum_info.get("stratum", ""))
            req_s = int(stratum_info.get("requested", -1))
            if req_s < 0 or "|gen=" not in stratum_key or "|lbl=" not in stratum_key:
                continue
            parts = dict(p.split("=", 1) for p in stratum_key.split("|")[1:] if "=" in p)
            if "lbl" in parts and "gen" in parts and "cls" not in parts:
                lbl_v = int(parts["lbl"])
                gen_v = parts["gen"]
                act_s = int(
                    ((sub["label"] == lbl_v) & (sub["source_generator_group"] == gen_v)).sum()
                )
                if act_s != req_s:
                    raise FrozenSnapshotError(
                        f"Generator stratum quota mismatch for '{stratum_key}': "
                        f"expected {req_s}, got {act_s}."
                    )

    # 8. Cross-split exact duplicate / leakage check
    leakage = check_leakage_boundaries(
        selection_df, split_column="release_split", raise_on_violation=False
    )
    if not leakage.passed:
        raise FrozenSnapshotError(
            f"Frozen selection violates leakage boundaries: "
            f"{leakage.cross_split_group_count} cross-split groups, "
            f"{leakage.cross_label_group_count} cross-label groups."
        )

    # 9. Verify original Plan ID consistency (never re-sample or forge!)
    release_id = str(plan_meta.get("release_id") or resolved_cfg.get("release_id") or "")
    sampling_config_hash = str(
        plan_meta.get("config_hash")
        or freeze_meta.get("sampling_config_hash")
        or freeze_meta.get("config_hash")
        or ""
    )
    recorded_plan_id = str(
        plan_meta.get("plan_id")
        or freeze_meta.get("plan_id")
        or freeze_meta.get("sampling_plan_id")
        or ""
    )
    if not release_id or not sampling_config_hash or not recorded_plan_id:
        raise FrozenSnapshotError(
            "Frozen snapshot metadata missing release_id, config_hash, or plan_id."
        )

    recomputed_plan_id = _compute_plan_id(release_id, sampling_config_hash, selection_df)
    if recomputed_plan_id != recorded_plan_id:
        raise FrozenSnapshotError(
            f"Original Plan ID integrity check failed: plan.json records '{recorded_plan_id}', "
            f"but selection + sampling config hash yields '{recomputed_plan_id}'."
        )
    if expected_plan_id is not None and recorded_plan_id != expected_plan_id:
        raise FrozenSnapshotError(
            f"Expected Plan ID '{expected_plan_id}', found '{recorded_plan_id}'."
        )

    sampling_code_commit = (
        freeze_meta.get("sampling_code_commit")
        or freeze_meta.get("code_commit_sha")
        or freeze_meta.get("git_commit")
        or plan_meta.get("code_commit_sha")
    )

    sampling_provenance = {
        "sampling_plan_id": recorded_plan_id,
        "sampling_config_hash": sampling_config_hash,
        "sampling_code_commit": sampling_code_commit,
        "release_id": release_id,
        "source_id": actual_source_id,
        "source_revision": actual_source_revision,
        "protocol_version": str(
            plan_meta.get("protocol_version") or cfg_proto.get("protocol_version")
        ),
        "sampling_seed": int(plan_meta.get("sampling_seed", resolved_cfg.get("sampling_seed", 0))),
        "snapshot_location": str(snapshot_location),
        "freeze_timestamp_utc": freeze_meta.get("created_at_utc")
        or freeze_meta.get("frozen_at_utc")
        or plan_meta.get("created_at_utc"),
    }

    validation_report = {
        "valid": True,
        "total_records": int(len(selection_df)),
        "unique_source_shards": int(selection_df["source_shard"].dropna().nunique()),
        "split_counts": observed_split_counts,
        "artifact_checks": artifact_checks,
        "leakage_check": leakage.to_dict(),
        "plan_id_verified": True,
        "arrow_locators_verified": bool(verify_arrow_locators and is_arrow_selection),
    }

    return FrozenSnapshot(
        snapshot_location=str(snapshot_location),
        freeze_meta=freeze_meta,
        plan_meta=plan_meta,
        resolved_sampling_config=resolved_cfg,
        selection_df=selection_df,
        sampling_provenance=sampling_provenance,
        validation_report=validation_report,
    )


def build_materialization_provenance(config: ReleaseConfig) -> dict[str, Any]:
    """Capture runtime & materialization-specific provenance separate from sampling provenance."""
    return {
        "materialization_config_hash": config.config_hash,
        "materialization_code_commit": _detect_git_commit(),
        "materialization_timestamp_utc": datetime.now(UTC).isoformat(),
        "materialization_runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "pyarrow_version": pa.__version__,
            "source_type": config.source.source_type,
            "source_location": config.source.location,
            "storage_backend": config.storage.backend,
            "storage_mode": (
                config.storage.gdrive_mode if config.storage.backend == "google_drive" else "local"
            ),
            "shard_target_bytes": config.shard_target_bytes,
            "max_samples_per_shard": config.max_samples_per_shard,
            "materialization_workers": config.materialization_workers,
            "materialization_batch_size": config.materialization_batch_size,
            "working_dir": config.working_dir,
            "arrow_cache_dir": config.source.arrow_cache_dir,
            "arrow_max_cached_shards": config.source.arrow_max_cached_shards,
            "arrow_max_cache_bytes": config.source.arrow_max_cache_bytes,
        },
    }


def check_frozen_plan_compatibility(
    snapshot: FrozenSnapshot,
    config: ReleaseConfig,
    is_pilot: bool = False,
    raise_on_incompatible: bool = True,
) -> CompatibilityReport:
    """Validate immutable semantic alignment while permitting authorized runtime config changes.

    Immutable semantic fields (MUST match exactly):
    - `source_id`, `source_revision`, `label_mapping`, `real_generator_value`
    - `protocol_version`, `seen_generators`, `unseen_generators`, split quotas & split rules
    - `sampling_seed`
    - `release_id` (unless `is_pilot=True`)

    Authorized runtime/implementation differences (permitted & logged explicitly):
    - `source_type` (e.g., archive/notebook -> `huggingface_arrow`), `source.location`,
      Arrow cache settings, `storage`, `working_dir`, `materialization_workers`,
      `shard_target_bytes`, `max_samples_per_shard`, hash/audit toggles.
    """
    s_cfg = snapshot.resolved_sampling_config
    s_src = s_cfg.get("source", {})
    s_proto = s_cfg.get("protocol", {})
    s_prov = snapshot.sampling_provenance

    checks: dict[str, bool] = {}
    errors: list[str] = []

    def _check(name: str, ok: bool, msg: str) -> None:
        checks[name] = ok
        if not ok:
            errors.append(msg)

    if not is_pilot:
        _check(
            "release_id",
            config.release_id == s_prov["release_id"],
            f"release_id mismatch: frozen='{s_prov['release_id']}', config='{config.release_id}'.",
        )
    else:
        checks["release_id"] = True

    _check(
        "source_id",
        config.source.source_id == s_prov["source_id"],
        f"source_id mismatch: frozen='{s_prov['source_id']}', config='{config.source.source_id}'.",
    )
    _check(
        "source_revision",
        config.source.source_revision == s_prov["source_revision"],
        f"source_revision mismatch: frozen='{s_prov['source_revision']}', "
        f"config='{config.source.source_revision}'.",
    )
    _check(
        "sampling_seed",
        config.sampling_seed == int(s_prov["sampling_seed"]),
        f"sampling_seed mismatch: frozen={s_prov['sampling_seed']}, config={config.sampling_seed}.",
    )
    _check(
        "protocol_version",
        config.protocol.protocol_version == s_prov["protocol_version"],
        f"protocol_version mismatch: frozen='{s_prov['protocol_version']}', "
        f"config='{config.protocol.protocol_version}'.",
    )

    # Label mapping & REAL provenance value
    s_lbl_map = {str(k).lower(): int(v) for k, v in s_src.get("label_mapping", {}).items()}
    if s_lbl_map:
        c_lbl_map = {str(k).lower(): int(v) for k, v in config.source.label_mapping.items()}
        _check(
            "label_mapping",
            c_lbl_map == s_lbl_map,
            f"label_mapping mismatch: frozen={s_lbl_map}, config={c_lbl_map}.",
        )
    else:
        checks["label_mapping"] = True

    s_real_gen = str(s_src.get("real_generator_value", "none")).lower()
    _check(
        "real_generator_value",
        config.source.real_generator_value.lower() == s_real_gen,
        f"real_generator_value mismatch: frozen='{s_real_gen}', "
        f"config='{config.source.real_generator_value}'.",
    )

    # Seen / Unseen generator sets
    s_seen = tuple(s_proto.get("seen_generators", []))
    s_unseen = tuple(s_proto.get("unseen_generators", []))
    if s_seen:
        _check(
            "seen_generators",
            set(config.protocol.seen_generators) == set(s_seen),
            f"seen_generators mismatch: frozen={list(s_seen)}, "
            f"config={list(config.protocol.seen_generators)}.",
        )
    if s_unseen:
        _check(
            "unseen_generators",
            set(config.protocol.unseen_generators) == set(s_unseen),
            f"unseen_generators mismatch: frozen={list(s_unseen)}, "
            f"config={list(config.protocol.unseen_generators)}.",
        )

    # Split rules & quotas
    s_splits = s_proto.get("splits", {})
    splits_ok = True
    if s_splits:
        if set(config.protocol.splits.keys()) != set(s_splits.keys()):
            splits_ok = False
            errors.append(
                f"Protocol split keys mismatch: frozen={sorted(s_splits.keys())}, "
                f"config={sorted(config.protocol.splits.keys())}."
            )
        else:
            for sp_name, sp_frozen in s_splits.items():
                sp_cfg = config.protocol.splits[sp_name]
                if (
                    sp_cfg.real_quota != int(sp_frozen["real_quota"])
                    or sp_cfg.fake_quota != int(sp_frozen["fake_quota"])
                    or sp_cfg.generator_pool != str(sp_frozen["generator_pool"]).lower()
                    or set(sp_cfg.allowed_source_splits) != set(sp_frozen["allowed_source_splits"])
                    or bool(sp_cfg.frozen_across_scaling)
                    != bool(sp_frozen.get("frozen_across_scaling", sp_name != "train"))
                ):
                    splits_ok = False
                    errors.append(
                        f"Protocol split rule mismatch for '{sp_name}': "
                        f"frozen={sp_frozen}, config={sp_cfg.to_dict()}."
                    )
    _check("protocol_splits", splits_ok, "Protocol split rules diverged from frozen snapshot.")

    # Record authorized runtime/adapter/storage differences explicitly
    runtime_diffs: dict[str, dict[str, Any]] = {}
    for field_name, frozen_val, current_val in (
        ("source.source_type", s_src.get("source_type"), config.source.source_type),
        ("source.location", s_src.get("location"), config.source.location),
        ("storage.backend", s_cfg.get("storage", {}).get("backend"), config.storage.backend),
        ("working_dir", s_cfg.get("working_dir"), config.working_dir),
        (
            "materialization_workers",
            s_cfg.get("materialization_workers"),
            config.materialization_workers,
        ),
        ("shard_target_bytes", s_cfg.get("shard_target_bytes"), config.shard_target_bytes),
        ("max_samples_per_shard", s_cfg.get("max_samples_per_shard"), config.max_samples_per_shard),
    ):
        if frozen_val != current_val:
            runtime_diffs[field_name] = {"sampling": frozen_val, "materialization": current_val}

    report = CompatibilityReport(
        compatible=len(errors) == 0,
        is_pilot=is_pilot,
        sampling_provenance=snapshot.sampling_provenance,
        materialization_provenance=build_materialization_provenance(config),
        checked_immutable_fields=checks,
        authorized_runtime_differences=runtime_diffs,
        incompatibilities=errors,
    )

    if raise_on_incompatible and not report.compatible:
        raise FrozenPlanCompatibilityError(
            f"Materialization config is incompatible with frozen plan "
            f"'{snapshot.sampling_provenance['sampling_plan_id']}': {errors}"
        )

    return report


def build_pilot_selection(
    snapshot: FrozenSnapshot,
    sample_count: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a deterministic, representative small pilot subset from a frozen snapshot.

    Guarantees (whenever available in `snapshot.selection_df`):
    - Both REAL (0) and FAKE (1) records
    - Multiple `source_generator_group`s
    - Multiple `source_split`s and `release_split`s
    - Multiple `source_shard`s
    Never mutates or modifies any row values from the frozen selection!
    """
    if sample_count < 1:
        raise ValueError(f"Pilot sample_count must be >= 1, got {sample_count}.")

    df = snapshot.selection_df
    if sample_count >= len(df):
        return df.copy().reset_index(drop=True)

    # Order rows deterministically by keyed hash inside each (release_split, label, source_generator_group, source_shard)
    def _rank_row(img_id: str) -> str:
        return hashlib.sha256(f"pilot|{seed}|{img_id}".encode()).hexdigest()

    work = df.copy()
    work["_pilot_rank"] = [_rank_row(str(iid)) for iid in work["image_id"].tolist()]
    work = work.sort_values(by=["_pilot_rank", "image_id"], kind="mergesort")

    # Round-robin pick across distinct (release_split, label, source_generator_group) buckets,
    # preferring new source_shards first so multi-shard behavior is exercised!
    buckets: dict[tuple[str, int, str], list[int]] = {}
    for idx, row in work.iterrows():
        key = (str(row["release_split"]), int(row["label"]), str(row["source_generator_group"]))
        buckets.setdefault(key, []).append(int(idx))

    bucket_keys = sorted(buckets.keys())
    selected_indices: list[int] = []
    seen_shards: set[str] = set()

    # Pass 1: diversify across buckets AND prefer unseen source_shards
    while len(selected_indices) < sample_count:
        progressed = False
        for key in bucket_keys:
            if len(selected_indices) >= sample_count:
                break
            candidates = buckets[key]
            if not candidates:
                continue
            # Prefer a candidate from an unseen source_shard when available
            pick_pos = 0
            for pos, r_idx in enumerate(candidates):
                sh = str(work.loc[r_idx, "source_shard"])
                if sh not in seen_shards:
                    pick_pos = pos
                    break
            chosen_idx = candidates.pop(pick_pos)
            selected_indices.append(chosen_idx)
            seen_shards.add(str(work.loc[chosen_idx, "source_shard"]))
            progressed = True
        if not progressed:
            break

    pilot_df = (
        df.loc[selected_indices]
        .sort_values(by=["release_split", "image_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return pilot_df
