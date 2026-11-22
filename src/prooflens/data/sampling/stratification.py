"""Deterministic group-aware stratified ordering and confounding risk diagnostics.

Provides:
- Stable cryptographic ranking keyed by `(seed, protocol_version, stratum_key, leakage_group_id)`
  that is strictly invariant to filesystem order and target release size (enabling nested scaling).
- Balanced quota distribution across generator groups (and optional ImageNet classes).
- Group-aware selection obeying explicit conflict policies (`exact_or_fail`,
  `skip_oversized_group`, `allow_underfill`) without ever splitting a leakage group.
- Confounding risk diagnostics across file format, resolution, source split, and generator.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd


class QuotaInfeasibleError(RuntimeError):
    """Raised when a requested sampling quota cannot be satisfied under protocol constraints."""


def deterministic_group_rank_key(
    seed: int,
    protocol_version: str,
    stratum_key: str,
    leakage_group_id: str,
) -> str:
    """Compute a stable SHA-256 rank key independent of filesystem order and release size."""
    payload = f"{seed}|{protocol_version}|{stratum_key}|{leakage_group_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def distribute_quota_evenly(quota: int, strata: list[str]) -> dict[str, int]:
    """Distribute an integer quota deterministically and as evenly as possible across sorted strata."""
    if quota < 0:
        raise ValueError(f"Quota must be non-negative, got {quota}")
    if not strata:
        if quota > 0:
            raise QuotaInfeasibleError(f"Cannot allocate quota={quota} across 0 strata.")
        return {}
    ordered = sorted(strata)
    base, rem = divmod(quota, len(ordered))
    return {s: base + (1 if idx < rem else 0) for idx, s in enumerate(ordered)}


@dataclass(frozen=True)
class StratumSelectionResult:
    """Outcome of group-aware deterministic selection for one `(split, label, stratum)` bucket."""

    stratum_key: str
    requested_quota: int
    selected_image_ids: list[str]
    selected_leakage_groups: set[str]
    available_eligible_images: int
    skipped_oversized_groups: int
    shortfall: int


def select_groups_deterministically(
    stratum_df: pd.DataFrame,
    stratum_key: str,
    target_quota: int,
    seed: int,
    protocol_version: str,
    excluded_groups: set[str],
    locked_image_ids: set[str] | None = None,
    deduplicate_within_group: bool = True,
    group_quota_policy: str = "exact_or_fail",
) -> StratumSelectionResult:
    """Select images deterministically from `stratum_df` respecting leakage groups and nested prefixing.

    Invariants:
    - Any `locked_image_ids` (from a parent release) are included first, preserving exact
      nested containment `Parent ⊆ Child`.
    - Any leakage group in `excluded_groups` (already assigned to another split) is never chosen.
    - All remaining eligible leakage groups are ordered strictly by
      `deterministic_group_rank_key(seed, protocol_version, stratum_key, leakage_group_id)`.
    - If `deduplicate_within_group=True`, only the lexicographically first `image_id` per
      leakage group is selected (preventing duplicate images in the same split).
    - If `deduplicate_within_group=False`, a whole leakage group is added atomically or skipped
      according to `group_quota_policy` — never split across splits!
    """
    locked = locked_image_ids or set()
    if target_quota == 0 and not locked:
        return StratumSelectionResult(
            stratum_key=stratum_key,
            requested_quota=0,
            selected_image_ids=[],
            selected_leakage_groups=set(),
            available_eligible_images=len(stratum_df),
            skipped_oversized_groups=0,
            shortfall=0,
        )

    # Organize candidate rows by leakage_group_id
    group_to_ids: dict[str, list[str]] = defaultdict(list)
    for img_id, lg_id in zip(
        stratum_df["image_id"].astype(str).tolist(),
        stratum_df["leakage_group_id"].astype(str).tolist(),
        strict=True,
    ):
        if lg_id in excluded_groups:
            continue
        group_to_ids[lg_id].append(img_id)

    for lg_id in group_to_ids:
        group_to_ids[lg_id].sort()

    selected_ids: list[str] = []
    selected_groups: set[str] = set()

    # 1. Lock parent release selections first (guarantees nested containment)
    if locked:
        for lg_id in sorted(group_to_ids.keys()):
            members = group_to_ids[lg_id]
            locked_in_grp = [m for m in members if m in locked]
            if locked_in_grp:
                selected_ids.extend(locked_in_grp)
                selected_groups.add(lg_id)

    if len(selected_ids) > target_quota:
        raise QuotaInfeasibleError(
            f"Parent release locked {len(selected_ids)} images for stratum '{stratum_key}', "
            f"which exceeds target_quota={target_quota}."
        )

    # Count total available eligible items respecting deduplication setting
    available_count = len(selected_ids)
    remaining_groups: list[tuple[str, str, list[str]]] = []
    for lg_id, members in group_to_ids.items():
        if lg_id in selected_groups:
            continue
        chosen_members = [members[0]] if deduplicate_within_group else members
        available_count += len(chosen_members)
        rank = deterministic_group_rank_key(seed, protocol_version, stratum_key, lg_id)
        remaining_groups.append((rank, lg_id, chosen_members))

    # Deterministic order independent of filesystem and release size
    remaining_groups.sort(key=lambda item: (item[0], item[1]))

    skipped_oversized = 0
    for _rank, lg_id, members in remaining_groups:
        remaining_needed = target_quota - len(selected_ids)
        if remaining_needed <= 0:
            break

        if len(members) <= remaining_needed:
            selected_ids.extend(members)
            selected_groups.add(lg_id)
        else:
            # Group-quota conflict: adding whole group would exceed quota, splitting violates leakage!
            skipped_oversized += 1
            if group_quota_policy in {"exact_or_fail", "skip_oversized_group"}:
                continue
            if group_quota_policy == "allow_underfill":
                break

    shortfall = max(0, target_quota - len(selected_ids))
    if shortfall > 0 and group_quota_policy != "allow_underfill":
        raise QuotaInfeasibleError(
            f"Infeasible quota for stratum '{stratum_key}': requested={target_quota}, "
            f"selected={len(selected_ids)}, shortfall={shortfall}, "
            f"available_eligible={available_count}, skipped_oversized_groups={skipped_oversized}, "
            f"policy='{group_quota_policy}'."
        )

    return StratumSelectionResult(
        stratum_key=stratum_key,
        requested_quota=target_quota,
        selected_image_ids=selected_ids,
        selected_leakage_groups=selected_groups,
        available_eligible_images=available_count,
        skipped_oversized_groups=skipped_oversized,
        shortfall=shortfall,
    )


def analyze_confounding_risks(df: pd.DataFrame) -> dict[str, Any]:
    """Detect and report confounding risks across format, resolution, source split, and generator."""
    if df.empty:
        return {
            "format_confounding_detected": False,
            "resolution_confounding_detected": False,
            "warnings": [],
            "format_by_label": {},
            "source_split_by_release_split": {},
        }

    warnings: list[str] = []
    format_by_label: dict[str, dict[str, int]] = {}
    format_confounding = False

    if "image_format" in df.columns and df["image_format"].notna().any():
        for lbl, grp in df.groupby("label"):
            counts = {
                str(k): int(v) for k, v in grp["image_format"].dropna().value_counts().items()
            }
            format_by_label[str(lbl)] = counts
        real_fmts = format_by_label.get("0", {})
        fake_fmts = format_by_label.get("1", {})
        if real_fmts and fake_fmts:
            top_real = max(real_fmts, key=real_fmts.get)
            top_fake = max(fake_fmts, key=fake_fmts.get)
            if top_real != top_fake:
                format_confounding = True
                warnings.append(
                    f"FORMAT_CONFOUNDING: REAL images are predominantly {top_real} ({real_fmts}) "
                    f"while FAKE images are predominantly {top_fake} ({fake_fmts})."
                )

    resolution_confounding = False
    if (
        "width" in df.columns
        and "height" in df.columns
        and df["width"].notna().any()
        and df["height"].notna().any()
    ):
        valid_wh = df.dropna(subset=["width", "height"])
        real_wh = valid_wh[valid_wh["label"] == 0]
        fake_wh = valid_wh[valid_wh["label"] == 1]
        if not real_wh.empty and not fake_wh.empty:
            fake_res = (
                fake_wh["width"].astype(int).astype(str)
                + "x"
                + fake_wh["height"].astype(int).astype(str)
            )
            real_res = (
                real_wh["width"].astype(int).astype(str)
                + "x"
                + real_wh["height"].astype(int).astype(str)
            )
            fake_top_ratio = fake_res.value_counts().iloc[0] / len(fake_res)
            real_top_ratio = real_res.value_counts().iloc[0] / len(real_res)
            if fake_top_ratio > 0.8 and real_top_ratio < 0.5:
                resolution_confounding = True
                warnings.append(
                    f"RESOLUTION_CONFOUNDING: {fake_top_ratio:.1%} of FAKE images share one fixed "
                    f"resolution ({fake_res.value_counts().index[0]}), whereas REAL images are "
                    f"heterogeneous (top resolution share {real_top_ratio:.1%})."
                )

    source_by_rel: dict[str, dict[str, int]] = {}
    if "release_split" in df.columns:
        for rsp, grp in df.groupby("release_split"):
            source_by_rel[str(rsp)] = {
                str(k): int(v) for k, v in grp["source_split"].value_counts().items()
            }

    return {
        "format_confounding_detected": format_confounding,
        "resolution_confounding_detected": resolution_confounding,
        "warnings": warnings,
        "format_by_label": format_by_label,
        "source_split_by_release_split": source_by_rel,
    }
