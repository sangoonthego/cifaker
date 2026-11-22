"""Deterministic nested release planner with frozen evaluation splits and replay verification.

Supports:
- Pre-materialization feasibility checking & dry-run planning
- Nested release scaling: Train(120K) ⊆ Train(200K) ⊆ Train(500K)
- Frozen evaluation splits (`dev`, `seen_test`, `unseen_test`) across scaling releases
- Parent release inheritance & strict protocol/seed compatibility checks
- Zero leakage across splits enforced by `LeakageGuard`
- Deterministic replacement planning when corrupted images are discovered during materialization
- Deterministic replay verification (`verify_plan_replay`)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from prooflens.data.config import ProtocolConfig, ReleaseConfig, SplitRuleConfig
from prooflens.data.registry.schema import (
    LABEL_FAKE,
    LABEL_REAL,
    read_canonical_parquet,
    validate_canonical_dataframe,
    write_canonical_parquet,
)
from prooflens.data.sampling.leakage_guard import (
    check_leakage_boundaries,
    filter_cross_source_split_leakage,
)
from prooflens.data.sampling.stratification import (
    QuotaInfeasibleError,
    analyze_confounding_risks,
    distribute_quota_evenly,
    select_groups_deterministically,
)
from prooflens.data.sources.base import SourceCapabilities


@dataclass(frozen=True)
class ReleasePlan:
    """Immutable release plan produced prior to image materialization."""

    plan_id: str
    release_id: str
    parent_release_id: str | None
    protocol_version: str
    sampling_seed: int
    config_hash: str
    feasible: bool
    selection_df: pd.DataFrame
    summary: dict[str, Any]

    def save(self, output_dir: Path | str) -> tuple[Path, Path]:
        """Persist plan summary JSON and frozen selection Parquet."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plan_json_path = out / "plan.json"
        selection_parquet_path = out / "plan_selection.parquet"

        plan_json_path.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        write_canonical_parquet(self.selection_df, selection_parquet_path)
        return plan_json_path, selection_parquet_path


def _split_allocation_order(splits: dict[str, SplitRuleConfig]) -> list[str]:
    """Allocate frozen evaluation splits FIRST so training expansion never perturbs them."""
    frozen = sorted(k for k, v in splits.items() if v.frozen_across_scaling)
    growing = sorted(k for k, v in splits.items() if not v.frozen_across_scaling)
    return frozen + growing


def _compute_plan_id(
    release_id: str,
    config_hash: str,
    selection_df: pd.DataFrame,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{release_id}|{config_hash}".encode())
    if not selection_df.empty:
        ordered = selection_df.sort_values(by=["release_split", "image_id"], kind="mergesort")
        for img_id, rsp in zip(
            ordered["image_id"].astype(str).tolist(),
            ordered["release_split"].astype(str).tolist(),
            strict=True,
        ):
            hasher.update(f"\n{rsp}:{img_id}".encode())
    return f"plan_{hasher.hexdigest()[:16]}"


def _select_for_split_and_label(
    eligible_df: pd.DataFrame,
    split_name: str,
    rule: SplitRuleConfig,
    label: int,
    target_quota: int,
    pool_generators: tuple[str, ...],
    protocol: ProtocolConfig,
    seed: int,
    reserved_groups: set[str],
    parent_locked_ids: set[str],
) -> tuple[list[str], set[str], list[dict[str, Any]]]:
    """Select images for one `(split_name, label)` respecting source splits, pools, and stratification."""
    sub = eligible_df[
        (eligible_df["label"] == label)
        & (eligible_df["source_split"].isin(rule.allowed_source_splits))
        & (eligible_df["source_generator_group"].isin(pool_generators))
    ]

    if target_quota == 0 and not parent_locked_ids:
        return [], set(), []

    selected_ids: list[str] = []
    newly_reserved: set[str] = set()
    stratum_reports: list[dict[str, Any]] = []

    if rule.stratify_by_generator_group:
        present_gens = sorted(set(pool_generators))
        gen_quotas = distribute_quota_evenly(target_quota, present_gens)
        for gen_group in present_gens:
            g_quota = gen_quotas[gen_group]
            g_df = sub[sub["source_generator_group"] == gen_group]
            g_locked = parent_locked_ids & set(g_df["image_id"].astype(str).tolist())

            if (
                rule.stratify_by_imagenet_class
                and "imagenet_class" in g_df.columns
                and g_df["imagenet_class"].notna().all()
                and not g_df.empty
            ):
                classes = sorted(g_df["imagenet_class"].astype(str).unique().tolist())
                cls_quotas = distribute_quota_evenly(g_quota, classes)
                for cls_name in classes:
                    c_df = g_df[g_df["imagenet_class"].astype(str) == cls_name]
                    c_locked = g_locked & set(c_df["image_id"].astype(str).tolist())
                    res = select_groups_deterministically(
                        stratum_df=c_df,
                        stratum_key=f"{split_name}|lbl={label}|gen={gen_group}|cls={cls_name}",
                        target_quota=cls_quotas[cls_name],
                        seed=seed,
                        protocol_version=protocol.protocol_version,
                        excluded_groups=reserved_groups | newly_reserved,
                        locked_image_ids=c_locked,
                        deduplicate_within_group=protocol.deduplicate_within_group,
                        group_quota_policy=protocol.group_quota_policy,
                    )
                    selected_ids.extend(res.selected_image_ids)
                    newly_reserved |= res.selected_leakage_groups
                    stratum_reports.append(
                        {
                            "stratum": res.stratum_key,
                            "requested": res.requested_quota,
                            "selected": len(res.selected_image_ids),
                            "available": res.available_eligible_images,
                            "shortfall": res.shortfall,
                        }
                    )
            else:
                res = select_groups_deterministically(
                    stratum_df=g_df,
                    stratum_key=f"{split_name}|lbl={label}|gen={gen_group}",
                    target_quota=g_quota,
                    seed=seed,
                    protocol_version=protocol.protocol_version,
                    excluded_groups=reserved_groups | newly_reserved,
                    locked_image_ids=g_locked,
                    deduplicate_within_group=protocol.deduplicate_within_group,
                    group_quota_policy=protocol.group_quota_policy,
                )
                selected_ids.extend(res.selected_image_ids)
                newly_reserved |= res.selected_leakage_groups
                stratum_reports.append(
                    {
                        "stratum": res.stratum_key,
                        "requested": res.requested_quota,
                        "selected": len(res.selected_image_ids),
                        "available": res.available_eligible_images,
                        "shortfall": res.shortfall,
                    }
                )
    else:
        res = select_groups_deterministically(
            stratum_df=sub,
            stratum_key=f"{split_name}|lbl={label}",
            target_quota=target_quota,
            seed=seed,
            protocol_version=protocol.protocol_version,
            excluded_groups=reserved_groups | newly_reserved,
            locked_image_ids=parent_locked_ids & set(sub["image_id"].astype(str).tolist()),
            deduplicate_within_group=protocol.deduplicate_within_group,
            group_quota_policy=protocol.group_quota_policy,
        )
        selected_ids.extend(res.selected_image_ids)
        newly_reserved |= res.selected_leakage_groups
        stratum_reports.append(
            {
                "stratum": res.stratum_key,
                "requested": res.requested_quota,
                "selected": len(res.selected_image_ids),
                "available": res.available_eligible_images,
                "shortfall": res.shortfall,
            }
        )

    return selected_ids, newly_reserved, stratum_reports


def plan_release(
    inventory_df: pd.DataFrame,
    config: ReleaseConfig,
    parent_manifest_df: pd.DataFrame | None = None,
    parent_release_meta: dict[str, Any] | None = None,
    source_capabilities: SourceCapabilities | None = None,
    dry_run: bool = False,
) -> ReleasePlan:
    """Create a deterministic, leakage-safe, nested release plan from a canonical inventory.

    Validates:
    - Upstream source-split leakage per `cross_source_split_leakage_policy`.
    - Parent release protocol version & seed match when expanding a parent release.
    - Exact preservation of parent's frozen evaluation splits (`dev`, `seen_test`, `unseen_test`)
      and strict prefix inheritance of parent's `train` split (`Parent_Train ⊆ Child_Train`).
    - Exact satisfaction of requested REAL/FAKE quotas (or explicit error with shortfall details).
    - Zero leakage groups crossing target splits in the resulting selection.
    """
    validated_inv = validate_canonical_dataframe(inventory_df)
    filtered_inv, upstream_leakage = filter_cross_source_split_leakage(
        validated_inv, policy=config.protocol.cross_source_split_leakage_policy
    )
    eligible = filtered_inv[filtered_inv["eligibility_status"] == "ELIGIBLE"].copy()

    # Validate parent release compatibility if expanding
    parent_by_split: dict[str, set[str]] = {}
    if config.parent_release_id is not None:
        if parent_manifest_df is None or parent_release_meta is None:
            raise ValueError(
                f"Release '{config.release_id}' references parent_release_id='{config.parent_release_id}', "
                "but parent manifest and metadata were not provided."
            )
        parent_proto = parent_release_meta.get("protocol_version")
        parent_seed = parent_release_meta.get("sampling_seed")
        if parent_proto != config.protocol.protocol_version:
            raise ValueError(
                f"Parent release protocol_version '{parent_proto}' does not match child "
                f"protocol_version '{config.protocol.protocol_version}'."
            )
        if parent_seed != config.sampling_seed:
            raise ValueError(
                f"Parent release sampling_seed {parent_seed} does not match child "
                f"sampling_seed {config.sampling_seed}."
            )
        if "release_split" not in parent_manifest_df.columns:
            raise ValueError("Parent manifest is missing required 'release_split' column.")

        eligible_ids_set = set(eligible["image_id"].astype(str).tolist())
        for sp_name, grp in parent_manifest_df.groupby("release_split"):
            sp_ids = set(grp["image_id"].astype(str).tolist())
            missing_from_inv = sp_ids - eligible_ids_set
            if missing_from_inv:
                raise QuotaInfeasibleError(
                    f"Parent release '{config.parent_release_id}' split '{sp_name}' has "
                    f"{len(missing_from_inv)} image_ids missing or excluded in current inventory."
                )
            parent_by_split[str(sp_name)] = sp_ids

    protocol = config.protocol
    reserved_groups: set[str] = set()
    all_selected_rows: list[pd.DataFrame] = []
    missing_quotas: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}

    allocation_order = _split_allocation_order(protocol.splits)
    eligible_indexed = eligible.set_index("image_id", drop=False)

    for split_name in allocation_order:
        rule = protocol.splits[split_name]
        pool_gens = (
            protocol.seen_generators
            if rule.generator_pool == "seen"
            else protocol.unseen_generators
        )
        parent_ids_for_split = parent_by_split.get(split_name, set())

        # If split is frozen and parent exists, enforce exact count match with parent!
        if rule.frozen_across_scaling and parent_ids_for_split:
            parent_split_df = eligible_indexed.loc[sorted(parent_ids_for_split)]
            parent_real_n = int((parent_split_df["label"] == LABEL_REAL).sum())
            parent_fake_n = int((parent_split_df["label"] == LABEL_FAKE).sum())
            if parent_real_n != rule.real_quota or parent_fake_n != rule.fake_quota:
                raise QuotaInfeasibleError(
                    f"Frozen split '{split_name}' quotas (real={rule.real_quota}, fake={rule.fake_quota}) "
                    f"must match parent release '{config.parent_release_id}' "
                    f"(real={parent_real_n}, fake={parent_fake_n})."
                )

        split_selected_ids: list[str] = []
        split_strata: list[dict[str, Any]] = []
        for label_val, quota in ((LABEL_REAL, rule.real_quota), (LABEL_FAKE, rule.fake_quota)):
            ids_lbl, groups_lbl, strata_lbl = _select_for_split_and_label(
                eligible_df=eligible,
                split_name=split_name,
                rule=rule,
                label=label_val,
                target_quota=quota,
                pool_generators=pool_gens,
                protocol=protocol,
                seed=config.sampling_seed,
                reserved_groups=reserved_groups,
                parent_locked_ids=parent_ids_for_split,
            )
            split_selected_ids.extend(ids_lbl)
            reserved_groups |= groups_lbl
            split_strata.extend(strata_lbl)
            for srep in strata_lbl:
                if srep["shortfall"] > 0:
                    missing_quotas.append(srep)

        if split_selected_ids:
            chosen_df = eligible_indexed.loc[split_selected_ids].copy()
            chosen_df["release_split"] = split_name
            all_selected_rows.append(chosen_df)

        split_reports[split_name] = {
            "requested_real": rule.real_quota,
            "requested_fake": rule.fake_quota,
            "requested_total": rule.total_quota,
            "selected_total": len(split_selected_ids),
            "frozen_across_scaling": rule.frozen_across_scaling,
            "generator_pool": rule.generator_pool,
            "allowed_source_splits": list(rule.allowed_source_splits),
            "strata": split_strata,
        }

    if all_selected_rows:
        selection_df = pd.concat(all_selected_rows, ignore_index=True)
        selection_df = selection_df.sort_values(
            by=["release_split", "image_id"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        selection_df = eligible.iloc[0:0].copy()
        selection_df["release_split"] = pd.Series(dtype=str)

    # Final strict cross-split leakage check on planned selection
    release_leakage = check_leakage_boundaries(
        selection_df,
        split_column="release_split",
        raise_on_violation=True,
    )

    # Verify nested containment explicitly if parent was provided
    parent_inheritance_verified = None
    if parent_by_split:
        for sp_name, parent_ids in parent_by_split.items():
            child_ids = set(
                selection_df.loc[selection_df["release_split"] == sp_name, "image_id"]
                .astype(str)
                .tolist()
            )
            if not parent_ids.issubset(child_ids):
                raise QuotaInfeasibleError(
                    f"Nested inheritance violated for split '{sp_name}': "
                    f"{len(parent_ids - child_ids)} parent IDs missing from child."
                )
            if protocol.splits[sp_name].frozen_across_scaling and parent_ids != child_ids:
                raise QuotaInfeasibleError(
                    f"Frozen evaluation split '{sp_name}' diverged from parent release."
                )
        parent_inheritance_verified = True

    # Honest storage estimate (only when file sizes exist in inventory; never false precision)
    known_sizes = (
        selection_df["file_size_bytes"].dropna()
        if "file_size_bytes" in selection_df.columns
        else pd.Series(dtype=float)
    )
    if len(known_sizes) == len(selection_df) and len(selection_df) > 0:
        est_bytes: int | None = int(known_sizes.sum())
        est_note = "Exact sum of inventory file_size_bytes for all selected records (excludes ~512B/file TAR headers)."
    elif len(known_sizes) > 0 and len(selection_df) > 0:
        mean_sz = float(known_sizes.mean())
        est_bytes = int(mean_sz * len(selection_df))
        est_note = (
            f"Rough estimate extrapolated from {len(known_sizes)}/{len(selection_df)} records "
            "with known file_size_bytes."
        )
    else:
        est_bytes = None
        est_note = (
            "Storage size cannot be estimated accurately prior to materialization because "
            "inventory lacks file_size_bytes."
        )

    # Determine delta images requiring new materialization vs reused from parent
    parent_all_ids = set().union(*parent_by_split.values()) if parent_by_split else set()
    selected_all_ids = set(selection_df["image_id"].astype(str).tolist())
    delta_count = len(selected_all_ids - parent_all_ids)
    reused_count = len(selected_all_ids & parent_all_ids)

    plan_id = _compute_plan_id(config.release_id, config.config_hash, selection_df)
    confounding = analyze_confounding_risks(selection_df)

    # Available counts summary
    avail_by_split_label = {
        f"{sp}|{lbl}": int(cnt)
        for (sp, lbl), cnt in eligible.groupby(["source_split", "label"]).size().items()
    }
    selected_class_dist = {
        str(k): int(v) for k, v in selection_df["label"].value_counts().sort_index().items()
    }
    selected_gen_dist = {
        str(k): int(v)
        for k, v in selection_df["source_generator_group"].value_counts().sort_index().items()
    }
    selected_src_split_dist = {
        str(k): int(v) for k, v in selection_df["source_split"].value_counts().sort_index().items()
    }

    caps_dict = source_capabilities.to_dict() if source_capabilities is not None else None
    io_limitation = None
    if source_capabilities is not None and not source_capabilities.metadata_only_enumeration:
        io_limitation = (
            "Source does not support metadata-only enumeration; planning or materialization "
            "requires streaming full image payloads from upstream."
        )

    summary: dict[str, Any] = {
        "plan_id": plan_id,
        "release_id": config.release_id,
        "parent_release_id": config.parent_release_id,
        "parent_inheritance_verified": parent_inheritance_verified,
        "protocol_version": protocol.protocol_version,
        "sampling_seed": config.sampling_seed,
        "config_hash": config.config_hash,
        "dry_run": dry_run,
        "feasible": len(missing_quotas) == 0,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "requested_total": sum(r.total_quota for r in protocol.splits.values()),
        "selected_total": int(len(selection_df)),
        "reused_from_parent_count": reused_count,
        "new_materialization_required_count": delta_count,
        "available_eligible_total": int(len(eligible)),
        "available_by_source_split_and_label": avail_by_split_label,
        "selected_class_distribution": selected_class_dist,
        "selected_generator_group_distribution": selected_gen_dist,
        "selected_source_split_distribution": selected_src_split_dist,
        "splits": split_reports,
        "missing_quota_report": missing_quotas,
        "upstream_leakage_audit": upstream_leakage.to_dict(),
        "planned_release_leakage_audit": release_leakage.to_dict(),
        "confounding_risks": confounding,
        "estimated_raw_image_bytes": est_bytes,
        "storage_estimate_note": est_note,
        "source_capabilities": caps_dict,
        "source_io_limitation_note": io_limitation,
    }

    return ReleasePlan(
        plan_id=plan_id,
        release_id=config.release_id,
        parent_release_id=config.parent_release_id,
        protocol_version=protocol.protocol_version,
        sampling_seed=config.sampling_seed,
        config_hash=config.config_hash,
        feasible=len(missing_quotas) == 0,
        selection_df=selection_df,
        summary=summary,
    )


def verify_plan_replay(
    inventory_df: pd.DataFrame,
    config: ReleaseConfig,
    existing_plan_selection_path: Path | str,
    parent_manifest_df: pd.DataFrame | None = None,
    parent_release_meta: dict[str, Any] | None = None,
) -> bool:
    """Re-run planning deterministically and verify identical selected IDs and split assignments."""
    saved_df = read_canonical_parquet(existing_plan_selection_path)
    replayed = plan_release(
        inventory_df=inventory_df,
        config=config,
        parent_manifest_df=parent_manifest_df,
        parent_release_meta=parent_release_meta,
        dry_run=True,
    )
    a = saved_df[["release_split", "image_id"]].sort_values(
        by=["release_split", "image_id"], kind="mergesort"
    )
    b = replayed.selection_df[["release_split", "image_id"]].sort_values(
        by=["release_split", "image_id"], kind="mergesort"
    )
    return bool(
        len(a) == len(b)
        and (a["image_id"].to_numpy() == b["image_id"].to_numpy()).all()
        and (a["release_split"].to_numpy() == b["release_split"].to_numpy()).all()
    )


def plan_replacements_for_exclusions(
    inventory_df: pd.DataFrame,
    config: ReleaseConfig,
    excluded_image_ids: set[str],
    exclusion_reasons: dict[str, str] | None = None,
    parent_manifest_df: pd.DataFrame | None = None,
    parent_release_meta: dict[str, Any] | None = None,
) -> ReleasePlan:
    """Explicitly mark corrupted/excluded IDs in inventory and generate a newly validated plan.

    Never silently substitutes images behind the scenes; produces a fresh immutable `plan_id`
    reflecting the explicit exclusions so full provenance is preserved.
    """
    updated = inventory_df.copy()
    mask = updated["image_id"].astype(str).isin(excluded_image_ids)
    updated.loc[mask, "eligibility_status"] = "EXCLUDED"
    if exclusion_reasons:
        for img_id, reason in exclusion_reasons.items():
            updated.loc[updated["image_id"] == img_id, "exclusion_reason"] = reason
    else:
        updated.loc[mask, "exclusion_reason"] = "MATERIALIZATION_CORRUPTED_OR_UNREADABLE"

    return plan_release(
        inventory_df=updated,
        config=config,
        parent_manifest_df=parent_manifest_df,
        parent_release_meta=parent_release_meta,
        dry_run=False,
    )
