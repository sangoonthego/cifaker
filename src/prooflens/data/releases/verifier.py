"""Pre-publication and consumer integrity verification for ProofLens dataset releases."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prooflens.data.registry.schema import (
    LABEL_FAKE,
    LABEL_REAL,
    read_canonical_parquet,
)
from prooflens.data.sampling.leakage_guard import check_leakage_boundaries
from prooflens.data.storage.base import StorageBackend


class ReleaseVerificationError(RuntimeError):
    """Raised when a dataset release fails one or more mandatory verification checks."""


@dataclass(frozen=True)
class ReleaseVerificationReport:
    """Structured result from verifying a staged or published dataset release."""

    release_id: str
    passed: bool
    checks: dict[str, bool]
    errors: list[str]
    warnings: list[str]
    shard_count: int
    total_images: int
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_release(
    storage: StorageBackend,
    release_id: str,
    verify_tar_members: bool = False,
) -> ReleaseVerificationReport:
    """Verify all mandatory dataset release invariants prior to or after publication.

    Checks performed:
    1. Required metadata artifacts (`manifest.parquet`, `checksums.json`, `plan.json`, `audit_summary.json`) exist.
    2. `manifest.parquet` SHA-256 matches `checksums.json` and passes canonical + release schema validation.
    3. Every shard referenced in `checksums.json` exists in storage and matches `byte_size` & `sha256`.
    4. Every image in `manifest.parquet` references a verified shard and member counts reconcile exactly.
    5. Zero cross-split leakage groups or exact SHA-256 duplicates across release splits.
    6. Zero unseen-generator samples in `train` or `dev`.
    7. All configured split quotas in `plan.json` are satisfied (no unresolved missing quotas or exclusions).
    8. If `parent_release_id` is present, parent `train` IDs are a subset of child `train` IDs and
       frozen evaluation split IDs (`dev`, `seen_test`, `unseen_test`) match identically.
    """
    prefix = f"releases/{release_id}"
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []

    manifest_key = f"{prefix}/manifest.parquet"
    checksums_key = f"{prefix}/checksums.json"
    plan_key = f"{prefix}/plan.json"
    audit_key = f"{prefix}/audit_summary.json"

    for name, key in (
        ("manifest_exists", manifest_key),
        ("checksums_exists", checksums_key),
        ("plan_exists", plan_key),
        ("audit_exists", audit_key),
    ):
        ok = storage.exists(key)
        checks[name] = ok
        if not ok:
            errors.append(f"Missing required release artifact: '{key}'")

    if errors:
        return ReleaseVerificationReport(
            release_id=release_id,
            passed=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
            shard_count=0,
            total_images=0,
            split_counts={},
        )

    checksums = json.loads(storage.read_bytes(checksums_key).decode("utf-8"))
    plan_meta = json.loads(storage.read_bytes(plan_key).decode("utf-8"))
    audit_meta = json.loads(storage.read_bytes(audit_key).decode("utf-8"))

    # Check exclusions — a release with unresolved exclusions is invalid until replanned!
    excluded_count = int(audit_meta.get("excluded_count", 0))
    checks["zero_unresolved_exclusions"] = excluded_count == 0
    if excluded_count > 0:
        errors.append(
            f"Release has {excluded_count} excluded/corrupted image(s) without replacement replanning."
        )

    # Verify manifest checksum & load Parquet
    expected_manifest_sha = checksums.get("manifest_parquet", {}).get("sha256")
    manifest_sha_ok = storage.verify_integrity(manifest_key, expected_sha256=expected_manifest_sha)
    checks["manifest_checksum_valid"] = manifest_sha_ok
    if not manifest_sha_ok:
        errors.append(f"Manifest SHA-256 verification failed for '{manifest_key}'.")

    manifest_df: pd.DataFrame
    with tempfile.TemporaryDirectory() as tmp:
        local_manifest = Path(tmp) / "manifest.parquet"
        storage.download(manifest_key, local_manifest)
        try:
            manifest_df = read_canonical_parquet(
                local_manifest, validate=True, require_release_columns=True
            )
            checks["manifest_schema_valid"] = True
        except Exception as exc:
            checks["manifest_schema_valid"] = False
            errors.append(f"Manifest schema validation failed: {exc}")
            return ReleaseVerificationReport(
                release_id=release_id,
                passed=False,
                checks=checks,
                errors=errors,
                warnings=warnings,
                shard_count=0,
                total_images=0,
                split_counts={},
            )

    # Verify every referenced shard in storage
    shards_map: dict[str, dict[str, Any]] = checksums.get("shards", {})
    all_shards_ok = True
    total_shard_images = 0

    for shard_id, s_info in sorted(shards_map.items()):
        s_key = str(s_info["storage_key"])
        s_sha = str(s_info["sha256"])
        s_size = int(s_info["byte_size"])
        s_count = int(s_info["image_count"])
        total_shard_images += s_count

        ok_remote = storage.verify_integrity(s_key, expected_sha256=s_sha, expected_size=s_size)
        if not ok_remote:
            all_shards_ok = False
            errors.append(f"Shard '{shard_id}' at '{s_key}' failed remote integrity verification.")
            continue

        if verify_tar_members:
            from prooflens.data.processing.shard_writer import verify_tar_shard_file

            with tempfile.TemporaryDirectory() as tmp_tar:
                local_tar = Path(tmp_tar) / f"{shard_id}.tar"
                storage.download(s_key, local_tar, expected_sha256=s_sha)
                ok_tar, tar_msg = verify_tar_shard_file(
                    local_tar,
                    expected_sha256=s_sha,
                    expected_byte_size=s_size,
                    expected_image_count=s_count,
                )
                if not ok_tar:
                    all_shards_ok = False
                    errors.append(f"Shard '{shard_id}' TAR structure invalid: {tar_msg}")

    checks["all_shards_verified"] = all_shards_ok

    # Reconcile manifest rows vs shards
    manifest_shard_ids = set(manifest_df["shard_id"].astype(str).unique().tolist())
    missing_refs = manifest_shard_ids - set(shards_map.keys())
    count_reconciled = (len(manifest_df) == total_shard_images) and (len(missing_refs) == 0)
    checks["manifest_shard_reconciled"] = count_reconciled
    if not count_reconciled:
        errors.append(
            f"Manifest/shard reconciliation mismatch: manifest rows={len(manifest_df)}, "
            f"sum(shard.image_count)={total_shard_images}, unindexed shards={sorted(missing_refs)}"
        )

    # Check leakage boundaries across release_split
    leakage = check_leakage_boundaries(
        manifest_df, split_column="release_split", raise_on_violation=False
    )
    checks["zero_cross_split_leakage"] = leakage.passed
    if not leakage.passed:
        errors.append(
            f"Cross-split leakage detected in release manifest: "
            f"{leakage.cross_split_group_count} cross-split groups, "
            f"{leakage.cross_label_group_count} cross-label groups."
        )

    # Check split quotas against plan
    plan_splits: dict[str, dict[str, Any]] = plan_meta.get("splits", {})
    quotas_ok = True
    split_counts: dict[str, int] = {}
    for sp_name, sp_spec in sorted(plan_splits.items()):
        sub = manifest_df[manifest_df["release_split"] == sp_name]
        split_counts[sp_name] = int(len(sub))
        req_real = int(sp_spec["requested_real"])
        req_fake = int(sp_spec["requested_fake"])
        act_real = int((sub["label"] == LABEL_REAL).sum())
        act_fake = int((sub["label"] == LABEL_FAKE).sum())
        if act_real != req_real or act_fake != req_fake:
            quotas_ok = False
            errors.append(
                f"Split '{sp_name}' quota mismatch: expected (real={req_real}, fake={req_fake}), "
                f"got (real={act_real}, fake={act_fake})."
            )
    checks["quotas_satisfied"] = quotas_ok

    # Verify unseen generator isolation from train & dev
    resolved_cfg = audit_meta.get("resolved_config", {})
    unseen_gens = set(resolved_cfg.get("protocol", {}).get("unseen_generators", []))
    if unseen_gens:
        td_sub = manifest_df[manifest_df["release_split"].isin(["train", "dev"])]
        leaked_unseen = td_sub[td_sub["source_generator_group"].isin(unseen_gens)]
        unseen_isolated = leaked_unseen.empty
        checks["unseen_generators_isolated"] = unseen_isolated
        if not unseen_isolated:
            errors.append(
                f"Unseen generator sample(s) found in train/dev: {len(leaked_unseen)} rows."
            )
    else:
        checks["unseen_generators_isolated"] = True

    # Verify parent nested inheritance when parent_release_id is present
    parent_id = plan_meta.get("parent_release_id")
    if parent_id:
        parent_manifest_key = f"releases/{parent_id}/manifest.parquet"
        if not storage.exists(parent_manifest_key):
            checks["parent_nested_inheritance"] = False
            errors.append(f"Referenced parent manifest missing: '{parent_manifest_key}'")
        else:
            with tempfile.TemporaryDirectory() as tmp_p:
                local_parent = Path(tmp_p) / "parent_manifest.parquet"
                storage.download(parent_manifest_key, local_parent)
                parent_df = read_canonical_parquet(
                    local_parent, validate=False, require_release_columns=False
                )
            inheritance_ok = True
            for sp_name, p_grp in parent_df.groupby("release_split"):
                p_ids = set(p_grp["image_id"].astype(str).tolist())
                c_ids = set(
                    manifest_df.loc[manifest_df["release_split"] == sp_name, "image_id"]
                    .astype(str)
                    .tolist()
                )
                is_frozen = bool(
                    plan_splits.get(str(sp_name), {}).get("frozen_across_scaling", False)
                )
                if not p_ids.issubset(c_ids):
                    inheritance_ok = False
                    errors.append(
                        f"Parent split '{sp_name}' is not a subset of child split '{sp_name}'."
                    )
                if is_frozen and p_ids != c_ids:
                    inheritance_ok = False
                    errors.append(
                        f"Frozen evaluation split '{sp_name}' changed relative to parent '{parent_id}'."
                    )
            checks["parent_nested_inheritance"] = inheritance_ok

    passed = all(checks.values()) and len(errors) == 0
    return ReleaseVerificationReport(
        release_id=release_id,
        passed=passed,
        checks=checks,
        errors=errors,
        warnings=warnings,
        shard_count=len(shards_map),
        total_images=int(len(manifest_df)),
        split_counts=split_counts,
    )
