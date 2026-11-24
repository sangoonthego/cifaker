"""Release lifecycle manager (plan/build/verify/publish) and reusable dataset consumer.

Separates dataset building (`ReleaseManager`) cleanly from dataset consumption
(`ReleaseConsumer`, `load_release`):
- Builders run inventory -> plan -> materialize -> verify -> publish (writing `release.json` LAST).
- Consumers reject unpublished or unverified releases, reuse verified cached shards,
  and support both `FullLocalShardCache` and `BoundedShardCache` (LRU budget eviction).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from prooflens.data.config import ReleaseConfig
from prooflens.data.processing.materializer import MaterializationResult, materialize_release
from prooflens.data.processing.shard_writer import ShardReceipt, compute_file_sha256
from prooflens.data.registry.schema import read_canonical_parquet, write_canonical_parquet
from prooflens.data.releases.frozen_plan import (
    CompatibilityReport,
    FrozenSnapshot,
    build_materialization_provenance,
    build_pilot_selection,
    check_frozen_plan_compatibility,
    load_and_validate_frozen_snapshot,
)
from prooflens.data.releases.state import ReleaseState, ReleaseStateTracker
from prooflens.data.releases.verifier import (
    ReleaseVerificationError,
    ReleaseVerificationReport,
    verify_release,
)
from prooflens.data.sampling.planner import ReleasePlan, plan_release
from prooflens.data.sampling.stratification import analyze_confounding_risks
from prooflens.data.sources.base import SourceAdapter
from prooflens.data.storage.base import ImmutabilityError, StorageBackend


class UnpublishedReleaseError(RuntimeError):
    """Raised when a consumer attempts to load a release that is not PUBLISHED."""


def _detect_git_commit_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=3
        ).strip()
        return out or None
    except Exception:
        return None


class ReleaseManager:
    """Orchestrates planning, resumable materialization, verification, and publication."""

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    def load_published_release_metadata(self, release_id: str) -> dict[str, Any]:
        """Read `release.json` and require status == 'PUBLISHED'."""
        key = f"releases/{release_id}/release.json"
        if not self._storage.exists(key):
            raise UnpublishedReleaseError(
                f"Release '{release_id}' is not published (missing '{key}')."
            )
        meta = json.loads(self._storage.read_bytes(key).decode("utf-8"))
        if meta.get("status") != ReleaseState.PUBLISHED.value:
            raise UnpublishedReleaseError(
                f"Release '{release_id}' has status={meta.get('status')!r}, expected 'PUBLISHED'."
            )
        return meta

    def _load_parent_artifacts(
        self,
        parent_release_id: str,
    ) -> tuple[dict[str, Any], pd.DataFrame, list[ShardReceipt]]:
        parent_meta = self.load_published_release_metadata(parent_release_id)
        manifest_key = f"releases/{parent_release_id}/manifest.parquet"
        expected_sha = parent_meta.get("manifest_sha256")
        with tempfile.TemporaryDirectory() as tmp:
            local_manifest = Path(tmp) / "parent_manifest.parquet"
            self._storage.download(manifest_key, local_manifest, expected_sha256=expected_sha)
            parent_df = read_canonical_parquet(
                local_manifest, validate=True, require_release_columns=True
            )

        # Reconstruct parent ShardReceipts from shard inventory + parent manifest
        receipts: list[ShardReceipt] = []
        shards_meta = parent_meta.get("shards", {})
        for shard_id, s_info in sorted(shards_meta.items()):
            sub = parent_df[parent_df["shard_id"] == shard_id]
            members = [
                {
                    "image_id": str(row["image_id"]),
                    "member_name": str(row["shard_member_name"]),
                    "offset_data": int(row["shard_offset_data"]),
                    "byte_size": int(row["file_size_bytes"]),
                    "sha256": str(row["sha256"]),
                    "label": int(row["label"]),
                    "generator": str(row["generator"]),
                    "source_generator_group": str(row["source_generator_group"]),
                    "source_split": str(row["source_split"]),
                    "release_split": str(row["release_split"]),
                    "width": int(row["width"]) if pd.notna(row.get("width")) else None,
                    "height": int(row["height"]) if pd.notna(row.get("height")) else None,
                    "image_format": (
                        str(row["image_format"]) if pd.notna(row.get("image_format")) else None
                    ),
                }
                for _, row in sub.iterrows()
            ]
            receipts.append(
                ShardReceipt(
                    shard_id=str(shard_id),
                    release_id=parent_release_id,
                    origin_release_id=str(s_info.get("origin_release_id", parent_release_id)),
                    release_split=str(s_info["release_split"]),
                    storage_key=str(s_info["storage_key"]),
                    status="COMMITTED",
                    image_count=int(s_info["image_count"]),
                    byte_size=int(s_info["byte_size"]),
                    sha256=str(s_info["sha256"]),
                    created_at_utc=str(
                        s_info.get("created_at_utc", parent_meta.get("published_at_utc", ""))
                    ),
                    members=members,
                )
            )
        return parent_meta, parent_df, receipts

    def plan(
        self,
        inventory_df: pd.DataFrame,
        config: ReleaseConfig,
        adapter: SourceAdapter | None = None,
        dry_run: bool = False,
    ) -> ReleasePlan:
        """Compute and (unless `dry_run=True`) persist a frozen release plan."""
        if not dry_run and self._storage.is_committed_release(config.release_id):
            raise ImmutabilityError(
                f"Release '{config.release_id}' is already PUBLISHED and cannot be re-planned."
            )

        parent_meta = None
        parent_df = None
        if config.parent_release_id is not None:
            parent_meta, parent_df, _ = self._load_parent_artifacts(config.parent_release_id)

        caps = adapter.capabilities() if adapter is not None else None
        rel_plan = plan_release(
            inventory_df=inventory_df,
            config=config,
            parent_manifest_df=parent_df,
            parent_release_meta=parent_meta,
            source_capabilities=caps,
            dry_run=dry_run,
        )

        if not dry_run:
            work_dir = Path(config.working_dir) / config.release_id
            plan_json, selection_parquet = rel_plan.save(work_dir)
            self._storage.upload(
                plan_json, f"releases/{config.release_id}/plan.json", overwrite=True
            )
            self._storage.upload(
                selection_parquet,
                f"releases/{config.release_id}/plan_selection.parquet",
                overwrite=True,
            )
            tracker = ReleaseStateTracker(self._storage, config.release_id)
            tracker.transition(
                ReleaseState.PLANNED,
                plan_id=rel_plan.plan_id,
                config_hash=config.config_hash,
                parent_release_id=config.parent_release_id,
                note="Release plan frozen and uploaded.",
            )

        return rel_plan

    def build(
        self,
        inventory_df: pd.DataFrame,
        adapter: SourceAdapter,
        config: ReleaseConfig,
        resume: bool = True,
    ) -> MaterializationResult:
        """Execute resumable, idempotent materialization for a configured release.

        Idempotency:
        - If `release_id` is already `PUBLISHED` with identical `config_hash`, returns cleanly
          without duplicating shards or mutating published artifacts.
        - If `release_id` is already `PUBLISHED` with a DIFFERENT `config_hash`, raises
          `ImmutabilityError` so published releases can never be silently overwritten.
        """
        prefix = f"releases/{config.release_id}"
        tracker = ReleaseStateTracker(self._storage, config.release_id)

        if self._storage.is_committed_release(config.release_id):
            pub_meta = self.load_published_release_metadata(config.release_id)
            if pub_meta.get("config_hash") != config.config_hash:
                raise ImmutabilityError(
                    f"Release '{config.release_id}' is already PUBLISHED with a different "
                    f"config_hash ({pub_meta.get('config_hash')!r} != {config.config_hash!r}). "
                    "Create a new release_id instead of mutating a published release."
                )
            # Idempotent no-op when re-running identical published release with resume=True
            if resume:
                _, pub_df, pub_receipts = self._load_parent_artifacts(config.release_id)
                from prooflens.data.registry.duplicates import DuplicateAuditReport

                dummy_dup = DuplicateAuditReport(
                    audit_scope="already_published_idempotent_noop",
                    total_records_inspected=len(pub_df),
                    records_with_sha256=int(pub_df["sha256"].notna().sum()),
                    exact_sha256_duplicate_groups=0,
                    exact_sha256_duplicate_files=0,
                    cross_source_split_sha256_groups=0,
                    cross_label_sha256_groups=0,
                    pixel_hash_duplicate_groups=0,
                    perceptual_candidate_pairs=0,
                    perceptual_included_in_leakage_groups=config.protocol.include_perceptual_in_leakage_groups,
                    leakage_group_count=int(pub_df["leakage_group_id"].nunique()),
                    multi_member_leakage_groups=0,
                    sample_exact_groups=[],
                    sample_near_duplicate_candidates=[],
                )
                return MaterializationResult(
                    release_id=config.release_id,
                    manifest_df=pub_df,
                    receipts=pub_receipts,
                    reused_parent_shards=0,
                    resumed_existing_shards=len(pub_receipts),
                    newly_materialized_shards=0,
                    excluded_records=[],
                    duplicate_report=dummy_dup,
                    duration_seconds=0.0,
                    metrics={"idempotent_published_skip": True},
                )
            raise ImmutabilityError(
                f"Release '{config.release_id}' is already PUBLISHED; pass --resume for idempotent check."
            )

        # Single-writer advisory lock check
        lock_key = f"{prefix}/.writer.lock.json"
        if config.storage.single_writer_lock:
            lock_payload = json.dumps(
                {
                    "release_id": config.release_id,
                    "acquired_at_utc": datetime.now(UTC).isoformat(),
                    "config_hash": config.config_hash,
                },
                indent=2,
            ).encode("utf-8")
            self._storage.write_bytes(lock_payload, lock_key, overwrite=True)

        parent_meta = None
        parent_df = None
        parent_receipts = None
        if config.parent_release_id is not None:
            parent_meta, parent_df, parent_receipts = self._load_parent_artifacts(
                config.parent_release_id
            )

        # Load existing frozen plan on resume if config_hash matches, else create fresh plan
        plan_key = f"{prefix}/plan.json"
        sel_key = f"{prefix}/plan_selection.parquet"
        selection_df: pd.DataFrame
        plan_id: str

        if resume and self._storage.exists(plan_key) and self._storage.exists(sel_key):
            existing_plan = json.loads(self._storage.read_bytes(plan_key).decode("utf-8"))
            if existing_plan.get("config_hash") != config.config_hash:
                raise ValueError(
                    f"Cannot resume release '{config.release_id}': existing plan config_hash "
                    f"{existing_plan.get('config_hash')!r} != current {config.config_hash!r}."
                )
            plan_id = str(existing_plan["plan_id"])
            with tempfile.TemporaryDirectory() as tmp:
                local_sel = Path(tmp) / "plan_selection.parquet"
                self._storage.download(sel_key, local_sel)
                selection_df = read_canonical_parquet(local_sel)
        else:
            rel_plan = self.plan(inventory_df, config, adapter=adapter, dry_run=False)
            selection_df = rel_plan.selection_df
            plan_id = rel_plan.plan_id

        tracker.transition(
            ReleaseState.MATERIALIZING,
            plan_id=plan_id,
            config_hash=config.config_hash,
            parent_release_id=config.parent_release_id,
            note="Materialization started.",
        )

        try:
            mat_res = materialize_release(
                selection_df=selection_df,
                adapter=adapter,
                storage=self._storage,
                config=config,
                parent_receipts=parent_receipts,
                parent_manifest_df=parent_df,
                resume=resume,
            )
        except KeyboardInterrupt:
            tracker.transition(
                ReleaseState.INTERRUPTED,
                error_message="Interrupted by user or runtime disconnect.",
                note="Safe to resume with --resume.",
            )
            raise
        except Exception as exc:
            tracker.transition(
                ReleaseState.FAILED,
                error_message=f"{type(exc).__name__}: {exc}",
                note="Materialization failed.",
            )
            raise
        finally:
            if config.storage.single_writer_lock and not self._storage.is_committed_release(
                config.release_id
            ):
                self._storage.delete_uncommitted(lock_key)

        # Persist manifest.parquet, checksums.json, and audit_summary.json
        work_dir = Path(config.working_dir) / config.release_id
        work_dir.mkdir(parents=True, exist_ok=True)
        local_manifest = work_dir / "manifest.parquet"
        write_canonical_parquet(mat_res.manifest_df, local_manifest, require_release_columns=True)
        manifest_sha = compute_file_sha256(local_manifest)
        manifest_size = local_manifest.stat().st_size

        self._storage.upload(
            local_manifest,
            f"{prefix}/manifest.parquet",
            expected_sha256=manifest_sha,
            overwrite=True,
        )

        shards_dict = {
            r.shard_id: {
                "shard_id": r.shard_id,
                "origin_release_id": r.origin_release_id,
                "release_split": r.release_split,
                "storage_key": r.storage_key,
                "image_count": r.image_count,
                "byte_size": r.byte_size,
                "sha256": r.sha256,
                "created_at_utc": r.created_at_utc,
            }
            for r in sorted(mat_res.receipts, key=lambda x: (x.release_split, x.shard_id))
        }

        checksums_payload = {
            "release_id": config.release_id,
            "manifest_parquet": {
                "storage_key": f"{prefix}/manifest.parquet",
                "byte_size": manifest_size,
                "sha256": manifest_sha,
            },
            "shards": shards_dict,
        }
        self._storage.write_bytes(
            json.dumps(checksums_payload, indent=2).encode("utf-8"),
            f"{prefix}/checksums.json",
            overwrite=True,
        )

        confounding = analyze_confounding_risks(mat_res.manifest_df)
        audit_payload = {
            "release_id": config.release_id,
            "plan_id": plan_id,
            "config_hash": config.config_hash,
            "resolved_config": config.to_dict(),
            "source_provenance": {
                "source_id": adapter.source_id,
                "source_revision": adapter.source_revision,
                "capabilities": adapter.capabilities().to_dict(),
            },
            "materialization_metrics": mat_res.metrics,
            "excluded_count": len(mat_res.excluded_records),
            "excluded_records_sample": mat_res.excluded_records[:50],
            "duplicate_audit": mat_res.duplicate_report.to_dict(),
            "confounding_risks": confounding,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        self._storage.write_bytes(
            json.dumps(audit_payload, indent=2).encode("utf-8"),
            f"{prefix}/audit_summary.json",
            overwrite=True,
        )

        tracker.transition(
            ReleaseState.MATERIALIZED,
            plan_id=plan_id,
            config_hash=config.config_hash,
            parent_release_id=config.parent_release_id,
            committed_shards=len(mat_res.receipts),
            total_images=len(mat_res.manifest_df),
            excluded_images=len(mat_res.excluded_records),
            note="Materialization finished; awaiting verification.",
        )

        return mat_res

    def build_from_frozen_snapshot(
        self,
        snapshot: FrozenSnapshot | str | Path,
        adapter: SourceAdapter,
        config: ReleaseConfig,
        snapshot_storage: StorageBackend | None = None,
        resume: bool = True,
        strict_frozen_mode: bool = True,
        expected_total_records: int | None = None,
    ) -> tuple[MaterializationResult, CompatibilityReport]:
        """Materialize an externally frozen sampling snapshot without re-running sampling.

        Preserves original `sampling_plan_id`, `sampling_config_hash`, `sampling_code_commit`,
        and all frozen selection rows while recording `materialization_config_hash`,
        `materialization_code_commit`, and runtime environment separately.
        """
        loaded_snapshot = (
            snapshot
            if isinstance(snapshot, FrozenSnapshot)
            else load_and_validate_frozen_snapshot(
                snapshot_location=snapshot,
                storage=snapshot_storage,
                expected_total_records=expected_total_records,
            )
        )
        compat = check_frozen_plan_compatibility(
            loaded_snapshot, config, is_pilot=False, raise_on_incompatible=True
        )

        prefix = f"releases/{config.release_id}"
        tracker = ReleaseStateTracker(self._storage, config.release_id)
        sampling_prov = compat.sampling_provenance
        mat_prov = compat.materialization_provenance
        sampling_plan_id = str(sampling_prov["sampling_plan_id"])
        sampling_cfg_hash = str(sampling_prov["sampling_config_hash"])

        if self._storage.is_committed_release(config.release_id):
            pub_meta = self.load_published_release_metadata(config.release_id)
            pub_plan_id = pub_meta.get("sampling_plan_id") or pub_meta.get("plan_id")
            if pub_plan_id != sampling_plan_id:
                raise ImmutabilityError(
                    f"Release '{config.release_id}' is already PUBLISHED with plan_id "
                    f"'{pub_plan_id}', which differs from frozen plan_id '{sampling_plan_id}'."
                )
            if resume:
                _, pub_df, pub_receipts = self._load_parent_artifacts(config.release_id)
                from prooflens.data.registry.duplicates import DuplicateAuditReport

                dummy_dup = DuplicateAuditReport(
                    audit_scope="already_published_frozen_idempotent_noop",
                    total_records_inspected=len(pub_df),
                    records_with_sha256=int(pub_df["sha256"].notna().sum()),
                    exact_sha256_duplicate_groups=0,
                    exact_sha256_duplicate_files=0,
                    cross_source_split_sha256_groups=0,
                    cross_label_sha256_groups=0,
                    pixel_hash_duplicate_groups=0,
                    perceptual_candidate_pairs=0,
                    perceptual_included_in_leakage_groups=config.protocol.include_perceptual_in_leakage_groups,
                    leakage_group_count=int(pub_df["leakage_group_id"].nunique()),
                    multi_member_leakage_groups=0,
                    sample_exact_groups=[],
                    sample_near_duplicate_candidates=[],
                )
                return (
                    MaterializationResult(
                        release_id=config.release_id,
                        manifest_df=pub_df,
                        receipts=pub_receipts,
                        reused_parent_shards=0,
                        resumed_existing_shards=len(pub_receipts),
                        newly_materialized_shards=0,
                        excluded_records=[],
                        duplicate_report=dummy_dup,
                        duration_seconds=0.0,
                        metrics={"idempotent_published_skip": True},
                    ),
                    compat,
                )
            raise ImmutabilityError(
                f"Release '{config.release_id}' is already PUBLISHED; pass --resume for idempotent check."
            )

        # If resuming an in-progress frozen build, ensure the stored plan matches the exact frozen plan_id
        plan_key = f"{prefix}/plan.json"
        sel_key = f"{prefix}/plan_selection.parquet"
        if resume and self._storage.exists(plan_key):
            existing_plan = json.loads(self._storage.read_bytes(plan_key).decode("utf-8"))
            if existing_plan.get("plan_id") != sampling_plan_id:
                raise ValueError(
                    f"Cannot resume release '{config.release_id}': existing plan_id "
                    f"{existing_plan.get('plan_id')!r} != frozen plan_id {sampling_plan_id!r}."
                )
        else:
            # Stage original immutable plan.json & plan_selection.parquet without mutating sampling provenance
            work_dir = Path(config.working_dir) / config.release_id
            work_dir.mkdir(parents=True, exist_ok=True)
            local_plan = work_dir / "plan.json"
            local_sel = work_dir / "plan_selection.parquet"
            local_plan.write_text(json.dumps(loaded_snapshot.plan_meta, indent=2), encoding="utf-8")
            write_canonical_parquet(loaded_snapshot.selection_df, local_sel)
            self._storage.upload(local_plan, plan_key, overwrite=True)
            self._storage.upload(local_sel, sel_key, overwrite=True)
            tracker.transition(
                ReleaseState.PLANNED,
                plan_id=sampling_plan_id,
                config_hash=sampling_cfg_hash,
                parent_release_id=config.parent_release_id,
                note="Imported externally frozen sampling plan (no re-sampling).",
            )

        # Persist separated materialization provenance & compatibility report
        self._storage.write_bytes(
            json.dumps(compat.to_dict(), indent=2).encode("utf-8"),
            f"{prefix}/materialization_provenance.json",
            overwrite=True,
        )

        lock_key = f"{prefix}/.writer.lock.json"
        if config.storage.single_writer_lock:
            lock_payload = json.dumps(
                {
                    "release_id": config.release_id,
                    "acquired_at_utc": datetime.now(UTC).isoformat(),
                    "sampling_plan_id": sampling_plan_id,
                    "materialization_config_hash": mat_prov["materialization_config_hash"],
                },
                indent=2,
            ).encode("utf-8")
            self._storage.write_bytes(lock_payload, lock_key, overwrite=True)

        tracker.transition(
            ReleaseState.MATERIALIZING,
            plan_id=sampling_plan_id,
            config_hash=sampling_cfg_hash,
            parent_release_id=config.parent_release_id,
            note="Frozen plan materialization started.",
        )

        try:
            mat_res = materialize_release(
                selection_df=loaded_snapshot.selection_df,
                adapter=adapter,
                storage=self._storage,
                config=config,
                parent_receipts=None,
                parent_manifest_df=None,
                resume=resume,
                strict_frozen_mode=strict_frozen_mode,
            )
        except KeyboardInterrupt:
            tracker.transition(
                ReleaseState.INTERRUPTED,
                error_message="Interrupted by user or runtime disconnect.",
                note="Safe to resume with --resume.",
            )
            raise
        except Exception as exc:
            tracker.transition(
                ReleaseState.FAILED,
                error_message=f"{type(exc).__name__}: {exc}",
                note="Frozen plan materialization failed.",
            )
            raise
        finally:
            if config.storage.single_writer_lock and not self._storage.is_committed_release(
                config.release_id
            ):
                self._storage.delete_uncommitted(lock_key)

        work_dir = Path(config.working_dir) / config.release_id
        work_dir.mkdir(parents=True, exist_ok=True)
        local_manifest = work_dir / "manifest.parquet"
        write_canonical_parquet(mat_res.manifest_df, local_manifest, require_release_columns=True)
        manifest_sha = compute_file_sha256(local_manifest)
        manifest_size = local_manifest.stat().st_size

        self._storage.upload(
            local_manifest,
            f"{prefix}/manifest.parquet",
            expected_sha256=manifest_sha,
            overwrite=True,
        )

        shards_dict = {
            r.shard_id: {
                "shard_id": r.shard_id,
                "origin_release_id": r.origin_release_id,
                "release_split": r.release_split,
                "storage_key": r.storage_key,
                "image_count": r.image_count,
                "byte_size": r.byte_size,
                "sha256": r.sha256,
                "created_at_utc": r.created_at_utc,
            }
            for r in sorted(mat_res.receipts, key=lambda x: (x.release_split, x.shard_id))
        }

        checksums_payload = {
            "release_id": config.release_id,
            "manifest_parquet": {
                "storage_key": f"{prefix}/manifest.parquet",
                "byte_size": manifest_size,
                "sha256": manifest_sha,
            },
            "shards": shards_dict,
        }
        self._storage.write_bytes(
            json.dumps(checksums_payload, indent=2).encode("utf-8"),
            f"{prefix}/checksums.json",
            overwrite=True,
        )

        confounding = analyze_confounding_risks(mat_res.manifest_df)
        audit_payload = {
            "release_id": config.release_id,
            "plan_id": sampling_plan_id,
            "config_hash": sampling_cfg_hash,
            "sampling_provenance": sampling_prov,
            "materialization_provenance": mat_prov,
            "compatibility_report": compat.to_dict(),
            "resolved_config": config.to_dict(),
            "resolved_sampling_config": loaded_snapshot.resolved_sampling_config,
            "source_provenance": {
                "source_id": adapter.source_id,
                "source_revision": adapter.source_revision,
                "capabilities": adapter.capabilities().to_dict(),
            },
            "materialization_metrics": mat_res.metrics,
            "excluded_count": len(mat_res.excluded_records),
            "excluded_records_sample": mat_res.excluded_records[:50],
            "duplicate_audit": mat_res.duplicate_report.to_dict(),
            "confounding_risks": confounding,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        self._storage.write_bytes(
            json.dumps(audit_payload, indent=2).encode("utf-8"),
            f"{prefix}/audit_summary.json",
            overwrite=True,
        )

        tracker.transition(
            ReleaseState.MATERIALIZED,
            plan_id=sampling_plan_id,
            config_hash=sampling_cfg_hash,
            parent_release_id=config.parent_release_id,
            committed_shards=len(mat_res.receipts),
            total_images=len(mat_res.manifest_df),
            excluded_images=len(mat_res.excluded_records),
            note="Frozen plan materialization finished; awaiting verification.",
        )

        return mat_res, compat

    def build_pilot_from_frozen_snapshot(
        self,
        snapshot: FrozenSnapshot | str | Path,
        adapter: SourceAdapter,
        config: ReleaseConfig,
        pilot_storage: StorageBackend,
        sample_count: int = 32,
        pilot_seed: int = 42,
        pilot_cache_dir: Path | str | None = None,
        snapshot_storage: StorageBackend | None = None,
    ) -> dict[str, Any]:
        """Run an isolated bounded pilot materialization into scratch `pilot_storage`.

        Pilot outputs are written exclusively to `pilot_storage` under a `pilot-` release ID
        with `is_pilot=True, official_release=False` so they can never be mistaken for or
        pollute an official production dataset release.
        """
        loaded_snapshot = (
            snapshot
            if isinstance(snapshot, FrozenSnapshot)
            else load_and_validate_frozen_snapshot(
                snapshot_location=snapshot,
                storage=snapshot_storage,
            )
        )
        compat = check_frozen_plan_compatibility(
            loaded_snapshot, config, is_pilot=True, raise_on_incompatible=True
        )

        pilot_df = build_pilot_selection(
            loaded_snapshot, sample_count=sample_count, seed=pilot_seed
        )
        pilot_release_id = f"pilot-{config.release_id}-n{len(pilot_df)}-s{pilot_seed}"

        from dataclasses import replace

        pilot_work = str(Path(config.working_dir) / "pilots" / pilot_release_id)
        pilot_cfg = replace(
            config,
            release_id=pilot_release_id,
            working_dir=pilot_work,
            strict_frozen_mode=True,
        )

        mat_res = materialize_release(
            selection_df=pilot_df,
            adapter=adapter,
            storage=pilot_storage,
            config=pilot_cfg,
            resume=False,
            strict_frozen_mode=True,
        )

        # Build pilot plan metadata matching actual pilot split counts so verifier validates exact pilot quotas
        pilot_splits_meta: dict[str, Any] = {}
        for sp_name, grp in pilot_df.groupby("release_split", sort=True):
            r_cnt = int((grp["label"] == 0).sum())
            f_cnt = int((grp["label"] == 1).sum())
            pilot_splits_meta[str(sp_name)] = {
                "requested_real": r_cnt,
                "requested_fake": f_cnt,
                "requested_total": len(grp),
                "selected_total": len(grp),
            }

        prefix = f"releases/{pilot_release_id}"
        pilot_plan_meta = {
            "is_pilot": True,
            "official_release": False,
            "pilot_release_id": pilot_release_id,
            "source_frozen_plan_id": compat.sampling_provenance["sampling_plan_id"],
            "source_frozen_release_id": compat.sampling_provenance["release_id"],
            "plan_id": compat.sampling_provenance["sampling_plan_id"],
            "release_id": pilot_release_id,
            "parent_release_id": None,
            "protocol_version": compat.sampling_provenance["protocol_version"],
            "sampling_seed": compat.sampling_provenance["sampling_seed"],
            "splits": pilot_splits_meta,
        }
        pilot_storage.write_bytes(
            json.dumps(pilot_plan_meta, indent=2).encode("utf-8"),
            f"{prefix}/plan.json",
            overwrite=True,
        )

        local_manifest = Path(pilot_work) / pilot_release_id / "manifest.parquet"
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_parquet(mat_res.manifest_df, local_manifest, require_release_columns=True)
        manifest_sha = compute_file_sha256(local_manifest)
        manifest_size = local_manifest.stat().st_size
        pilot_storage.upload(
            local_manifest,
            f"{prefix}/manifest.parquet",
            expected_sha256=manifest_sha,
            overwrite=True,
        )

        shards_dict = {
            r.shard_id: {
                "shard_id": r.shard_id,
                "origin_release_id": r.origin_release_id,
                "release_split": r.release_split,
                "storage_key": r.storage_key,
                "image_count": r.image_count,
                "byte_size": r.byte_size,
                "sha256": r.sha256,
                "created_at_utc": r.created_at_utc,
            }
            for r in sorted(mat_res.receipts, key=lambda x: (x.release_split, x.shard_id))
        }
        pilot_storage.write_bytes(
            json.dumps(
                {
                    "release_id": pilot_release_id,
                    "is_pilot": True,
                    "manifest_parquet": {
                        "storage_key": f"{prefix}/manifest.parquet",
                        "byte_size": manifest_size,
                        "sha256": manifest_sha,
                    },
                    "shards": shards_dict,
                },
                indent=2,
            ).encode("utf-8"),
            f"{prefix}/checksums.json",
            overwrite=True,
        )
        pilot_storage.write_bytes(
            json.dumps(
                {
                    "is_pilot": True,
                    "official_release": False,
                    "release_id": pilot_release_id,
                    "sampling_provenance": compat.sampling_provenance,
                    "materialization_provenance": build_materialization_provenance(pilot_cfg),
                    "resolved_config": pilot_cfg.to_dict(),
                    "materialization_metrics": mat_res.metrics,
                    "excluded_count": len(mat_res.excluded_records),
                    "duplicate_audit": mat_res.duplicate_report.to_dict(),
                },
                indent=2,
            ).encode("utf-8"),
            f"{prefix}/audit_summary.json",
            overwrite=True,
        )

        ver_report = verify_release(pilot_storage, pilot_release_id, verify_tar_members=True)

        # Read-back verification of every pilot image via TAR byte offsets
        cache_root = (
            Path(pilot_cache_dir)
            if pilot_cache_dir is not None
            else Path(pilot_work) / "pilot_readback_cache"
        )
        readback_verified = 0
        shard_local_map: dict[str, Path] = {}
        for sid, sinfo in shards_dict.items():
            dst = cache_root / f"{sid}.tar"
            pilot_storage.download(
                str(sinfo["storage_key"]), dst, expected_sha256=str(sinfo["sha256"])
            )
            shard_local_map[sid] = dst

        from prooflens.data.shard_dataset import ShardMapDataset

        ds = ShardMapDataset(
            manifest_df=mat_res.manifest_df,
            shard_paths=shard_local_map,
            transform=None,
            verify_member_sha256=True,
        )
        try:
            for i in range(len(ds)):
                sample = ds[i]
                assert sample["image"].size[0] > 0 and sample["image"].size[1] > 0
                readback_verified += 1
        finally:
            ds.close()

        pilot_report = {
            "is_pilot": True,
            "official_release": False,
            "pilot_release_id": pilot_release_id,
            "source_frozen_plan_id": compat.sampling_provenance["sampling_plan_id"],
            "pilot_samples_requested": sample_count,
            "pilot_samples_materialized": int(len(mat_res.manifest_df)),
            "pilot_readback_verified": readback_verified,
            "pilot_real_count": int((mat_res.manifest_df["label"] == 0).sum()),
            "pilot_fake_count": int((mat_res.manifest_df["label"] == 1).sum()),
            "pilot_generator_groups": sorted(
                mat_res.manifest_df["source_generator_group"].astype(str).unique().tolist()
            ),
            "pilot_source_splits": sorted(
                mat_res.manifest_df["source_split"].astype(str).unique().tolist()
            ),
            "pilot_release_splits": sorted(
                mat_res.manifest_df["release_split"].astype(str).unique().tolist()
            ),
            "pilot_unique_source_shards": int(
                mat_res.manifest_df["source_shard"].dropna().nunique()
            ),
            "pilot_output_shards": len(mat_res.receipts),
            "verification": ver_report.to_dict(),
            "metrics": mat_res.metrics,
            "sampling_provenance": compat.sampling_provenance,
            "materialization_provenance": build_materialization_provenance(pilot_cfg),
            "passed": bool(ver_report.passed and readback_verified == len(pilot_df)),
        }
        pilot_storage.write_bytes(
            json.dumps(pilot_report, indent=2).encode("utf-8"),
            f"{prefix}/pilot_report.json",
            overwrite=True,
        )
        return pilot_report

    def verify(
        self,
        release_id: str,
        verify_tar_members: bool = False,
    ) -> ReleaseVerificationReport:
        """Verify all invariants for a release and advance state to VERIFIED if passing."""
        report = verify_release(
            self._storage, release_id=release_id, verify_tar_members=verify_tar_members
        )
        if not self._storage.is_committed_release(release_id):
            tracker = ReleaseStateTracker(self._storage, release_id)
            if report.passed:
                tracker.transition(
                    ReleaseState.VERIFIED,
                    committed_shards=report.shard_count,
                    total_images=report.total_images,
                    note="All verification checks passed.",
                )
            else:
                tracker.transition(
                    ReleaseState.FAILED,
                    error_message="; ".join(report.errors),
                    note="Verification failed.",
                )
        return report

    def publish(
        self,
        release_id: str,
        verify_tar_members: bool = False,
    ) -> dict[str, Any]:
        """Verify release invariants and publish `release.json` LAST as the publication commit."""
        if self._storage.is_committed_release(release_id):
            raise ImmutabilityError(
                f"Release '{release_id}' is already PUBLISHED and cannot be re-published."
            )

        report = self.verify(release_id, verify_tar_members=verify_tar_members)
        if not report.passed:
            raise ReleaseVerificationError(
                f"Cannot publish release '{release_id}'; verification failed: {report.errors}"
            )

        prefix = f"releases/{release_id}"
        checksums = json.loads(self._storage.read_bytes(f"{prefix}/checksums.json").decode("utf-8"))
        plan_meta = json.loads(self._storage.read_bytes(f"{prefix}/plan.json").decode("utf-8"))
        audit_meta = json.loads(
            self._storage.read_bytes(f"{prefix}/audit_summary.json").decode("utf-8")
        )

        resolved_cfg = audit_meta.get("resolved_config", {})
        tracker = ReleaseStateTracker(self._storage, release_id)
        tracker.transition(
            ReleaseState.PUBLISHED,
            committed_shards=report.shard_count,
            total_images=report.total_images,
            note="Writing final immutable release.json.",
        )

        release_payload: dict[str, Any] = {
            "release_id": release_id,
            "status": ReleaseState.PUBLISHED.value,
            "published_at_utc": datetime.now(UTC).isoformat(),
            "plan_id": plan_meta.get("plan_id"),
            "parent_release_id": plan_meta.get("parent_release_id"),
            "protocol_version": plan_meta.get("protocol_version"),
            "sampling_seed": plan_meta.get("sampling_seed"),
            "sampling_policy": {
                "group_quota_policy": resolved_cfg.get("protocol", {}).get("group_quota_policy"),
                "cross_source_split_leakage_policy": resolved_cfg.get("protocol", {}).get(
                    "cross_source_split_leakage_policy"
                ),
                "deduplicate_within_group": resolved_cfg.get("protocol", {}).get(
                    "deduplicate_within_group"
                ),
            },
            "source_provenance": audit_meta.get("source_provenance", {}),
            "sampling_provenance": audit_meta.get("sampling_provenance"),
            "materialization_provenance": audit_meta.get("materialization_provenance"),
            "sampling_plan_id": (
                audit_meta.get("sampling_provenance", {}).get("sampling_plan_id")
                if isinstance(audit_meta.get("sampling_provenance"), dict)
                else plan_meta.get("plan_id")
            ),
            "sampling_config_hash": (
                audit_meta.get("sampling_provenance", {}).get("sampling_config_hash")
                if isinstance(audit_meta.get("sampling_provenance"), dict)
                else audit_meta.get("config_hash")
            ),
            "materialization_config_hash": (
                audit_meta.get("materialization_provenance", {}).get("materialization_config_hash")
                if isinstance(audit_meta.get("materialization_provenance"), dict)
                else audit_meta.get("config_hash")
            ),
            "config_hash": audit_meta.get("config_hash"),
            "resolved_config": resolved_cfg,
            "code_commit_sha": _detect_git_commit_sha(),
            "manifest_key": f"{prefix}/manifest.parquet",
            "manifest_sha256": checksums["manifest_parquet"]["sha256"],
            "manifest_byte_size": checksums["manifest_parquet"]["byte_size"],
            "checksums_key": f"{prefix}/checksums.json",
            "audit_summary_key": f"{prefix}/audit_summary.json",
            "split_counts": report.split_counts,
            "total_images": report.total_images,
            "shard_count": report.shard_count,
            "shards": checksums["shards"],
            "verification_checks": report.checks,
            "audit_summary": {
                "duplicate_audit": audit_meta.get("duplicate_audit"),
                "confounding_risks": audit_meta.get("confounding_risks"),
                "materialization_metrics": audit_meta.get("materialization_metrics"),
            },
            "storage_semantics_note": (
                f"Published on {self._storage.backend_name} using single-writer metadata-last "
                "commit (release.json written last; non-transactional across multiple files on Drive)."
            ),
        }

        # Metadata-last publication commit
        self._storage.write_bytes(
            json.dumps(release_payload, indent=2).encode("utf-8"),
            f"{prefix}/release.json",
            overwrite=False,
        )
        return release_payload


# ----------------------------------------------------------------------
# Dataset Consumer & Shard Caches (Full Local Cache + Bounded LRU Cache)
# ----------------------------------------------------------------------


class ShardCacheManager:
    """Base shard cache interface staging verified shards locally for model consumers."""

    def __init__(self, cache_dir: Path | str, storage: StorageBackend) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._storage = storage
        self.cache_hits = 0
        self.cache_misses = 0

    def local_shard_path(self, storage_key: str) -> Path:
        return self.cache_dir / "shards" / storage_key

    def ensure_shard(
        self,
        storage_key: str,
        expected_sha256: str,
        expected_size: int | None = None,
        verify_cached_checksum: bool = True,
    ) -> Path:
        """Return local path to a verified shard, reusing existing cached file when valid."""
        local_path = self.local_shard_path(storage_key)
        if local_path.is_file():
            size_ok = expected_size is None or local_path.stat().st_size == expected_size
            if size_ok and (
                not verify_cached_checksum or compute_file_sha256(local_path) == expected_sha256
            ):
                self.cache_hits += 1
                self._on_access(storage_key, local_path)
                return local_path

        self.cache_misses += 1
        self._before_download(expected_size or 0)
        self._storage.download(storage_key, local_path, expected_sha256=expected_sha256)
        self._on_access(storage_key, local_path)
        return local_path

    def _before_download(self, incoming_bytes: int) -> None:
        """Hook for bounded caches to evict old shards before staging a new shard."""

    def _on_access(self, storage_key: str, local_path: Path) -> None:
        """Hook to track shard access recency."""


class FullLocalShardCache(ShardCacheManager):
    """Stages all required shards for a split into local cache and reuses verified copies."""


class BoundedShardCache(ShardCacheManager):
    """LRU bounded-disk shard cache that caps local staged shard bytes for large releases."""

    def __init__(
        self,
        cache_dir: Path | str,
        storage: StorageBackend,
        max_cache_bytes: int,
    ) -> None:
        if max_cache_bytes <= 0:
            raise ValueError(f"max_cache_bytes must be positive, got {max_cache_bytes}")
        super().__init__(cache_dir=cache_dir, storage=storage)
        self.max_cache_bytes = int(max_cache_bytes)
        self._lru: OrderedDict[str, tuple[Path, int]] = OrderedDict()
        self.evictions = 0

    @property
    def current_bytes(self) -> int:
        return sum(sz for _, sz in self._lru.values())

    def _before_download(self, incoming_bytes: int) -> None:
        while self._lru and (self.current_bytes + incoming_bytes > self.max_cache_bytes):
            _old_key, (old_path, _old_sz) = self._lru.popitem(last=False)
            old_path.unlink(missing_ok=True)
            self.evictions += 1

    def _on_access(self, storage_key: str, local_path: Path) -> None:
        sz = local_path.stat().st_size if local_path.is_file() else 0
        if storage_key in self._lru:
            self._lru.move_to_end(storage_key)
        self._lru[storage_key] = (local_path, sz)


@dataclass(frozen=True)
class PreparedReleaseSplit:
    """Result of preparing a published release split in local cache."""

    release_id: str
    split: str
    release_metadata: dict[str, Any]
    manifest_df: pd.DataFrame
    shard_paths: dict[str, Path]
    cache_hits: int
    cache_misses: int
    prepare_seconds: float


class ReleaseConsumer:
    """Read-only consumer API for preparing and loading published dataset releases."""

    def __init__(self, storage: StorageBackend, cache_dir: Path | str) -> None:
        self._storage = storage
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_published_release(self, release_id: str) -> dict[str, Any]:
        """Fetch and validate `release.json`, rejecting unpublished or incomplete releases."""
        rel_key = f"releases/{release_id}/release.json"
        if not self._storage.exists(rel_key):
            raise UnpublishedReleaseError(
                f"Cannot consume release '{release_id}': '{rel_key}' does not exist (unpublished)."
            )
        meta = json.loads(self._storage.read_bytes(rel_key).decode("utf-8"))
        if meta.get("status") != ReleaseState.PUBLISHED.value:
            raise UnpublishedReleaseError(
                f"Cannot consume release '{release_id}': status is {meta.get('status')!r}, not 'PUBLISHED'."
            )
        return meta

    def load_manifest(self, release_id: str, verify_checksum: bool = True) -> pd.DataFrame:
        """Download (or reuse cached) verified `manifest.parquet` for a published release."""
        meta = self.get_published_release(release_id)
        manifest_key = str(meta["manifest_key"])
        expected_sha = str(meta["manifest_sha256"])

        local_manifest = self._cache_dir / "releases" / release_id / "manifest.parquet"
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        need_dl = True
        if local_manifest.is_file():
            if not verify_checksum or compute_file_sha256(local_manifest) == expected_sha:
                need_dl = False
        if need_dl:
            self._storage.download(manifest_key, local_manifest, expected_sha256=expected_sha)

        return read_canonical_parquet(local_manifest, validate=True, require_release_columns=True)

    def prepare_split(
        self,
        release_id: str,
        split: str | None = None,
        verify_checksums: bool = True,
        max_cache_bytes: int | None = None,
    ) -> PreparedReleaseSplit:
        """Stage and verify all shards required for `split` (or all splits if `split=None`)."""
        t0 = time.perf_counter()
        meta = self.get_published_release(release_id)
        manifest_df = self.load_manifest(release_id, verify_checksum=verify_checksums)

        if split is not None:
            available_splits = sorted(manifest_df["release_split"].unique().tolist())
            if split not in available_splits:
                raise ValueError(
                    f"Split '{split}' not found in release '{release_id}'. "
                    f"Available splits: {available_splits}"
                )
            split_df = manifest_df[manifest_df["release_split"] == split].reset_index(drop=True)
        else:
            split_df = manifest_df.reset_index(drop=True)

        cache: ShardCacheManager = (
            BoundedShardCache(self._cache_dir, self._storage, max_cache_bytes=max_cache_bytes)
            if max_cache_bytes is not None
            else FullLocalShardCache(self._cache_dir, self._storage)
        )

        shards_meta = meta.get("shards", {})
        needed_shard_ids = sorted(split_df["shard_id"].astype(str).unique().tolist())
        shard_paths: dict[str, Path] = {}

        for shard_id in needed_shard_ids:
            if shard_id not in shards_meta:
                raise ReleaseVerificationError(
                    f"Manifest references shard '{shard_id}' missing from release.json."
                )
            s_info = shards_meta[shard_id]
            local_path = cache.ensure_shard(
                storage_key=str(s_info["storage_key"]),
                expected_sha256=str(s_info["sha256"]),
                expected_size=int(s_info["byte_size"]),
                verify_cached_checksum=verify_checksums,
            )
            shard_paths[shard_id] = local_path

        return PreparedReleaseSplit(
            release_id=release_id,
            split=split or "all",
            release_metadata=meta,
            manifest_df=split_df,
            shard_paths=shard_paths,
            cache_hits=cache.cache_hits,
            cache_misses=cache.cache_misses,
            prepare_seconds=time.perf_counter() - t0,
        )


def load_release(
    release: str,
    split: str,
    cache_dir: Path | str,
    storage: StorageBackend,
    transform: Callable[..., Any] | None = None,
    verify_checksums: bool = True,
    streaming: bool = False,
    max_cache_bytes: int | None = None,
    seed: int = 42,
    shuffle_shards: bool = False,
) -> Any:
    """High-level consumer API returning a PyTorch-compatible Dataset backed by cached TAR shards.

    Example:
        dataset = load_release(
            release="genimage-120k-v1",
            split="train",
            cache_dir="/content/prooflens_cache",
            storage=storage_backend,
            transform=my_model_transform,
        )
    """
    from prooflens.data.shard_dataset import ShardIterableDataset, ShardMapDataset

    consumer = ReleaseConsumer(storage=storage, cache_dir=cache_dir)
    if streaming and max_cache_bytes is not None:
        meta = consumer.get_published_release(release)
        manifest_df = consumer.load_manifest(release, verify_checksum=verify_checksums)
        split_df = manifest_df[manifest_df["release_split"] == split].reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"Split '{split}' is empty or missing in release '{release}'.")
        cache = BoundedShardCache(cache_dir, storage, max_cache_bytes=max_cache_bytes)
        return ShardIterableDataset(
            manifest_df=split_df,
            shards_metadata=meta["shards"],
            shard_cache=cache,
            transform=transform,
            seed=seed,
            shuffle_shards=shuffle_shards,
            verify_cached_checksum=verify_checksums,
        )

    prepared = consumer.prepare_split(
        release_id=release,
        split=split,
        verify_checksums=verify_checksums,
    )
    if streaming:
        cache = FullLocalShardCache(cache_dir, storage)
        return ShardIterableDataset(
            manifest_df=prepared.manifest_df,
            shards_metadata=prepared.release_metadata["shards"],
            shard_cache=cache,
            transform=transform,
            seed=seed,
            shuffle_shards=shuffle_shards,
            verify_cached_checksum=verify_checksums,
        )

    return ShardMapDataset(
        manifest_df=prepared.manifest_df,
        shard_paths=prepared.shard_paths,
        transform=transform,
    )
