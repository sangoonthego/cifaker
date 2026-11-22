"""Leakage guard enforcing split isolation and detecting cross-boundary duplicate groups."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd


class LeakageViolationError(RuntimeError):
    """Raised when a leakage group crosses protected dataset split boundaries."""


@dataclass(frozen=True)
class LeakageCheckResult:
    """Result of auditing leakage groups across source or release splits."""

    passed: bool
    split_column: str
    cross_split_group_count: int
    cross_label_group_count: int
    cross_split_groups: list[dict[str, Any]]
    cross_label_groups: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "split_column": self.split_column,
            "cross_split_group_count": self.cross_split_group_count,
            "cross_label_group_count": self.cross_label_group_count,
            "cross_split_groups": self.cross_split_groups,
            "cross_label_groups": self.cross_label_groups,
        }


def check_leakage_boundaries(
    df: pd.DataFrame,
    split_column: str = "release_split",
    group_column: str = "leakage_group_id",
    sha256_column: str = "sha256",
    raise_on_violation: bool = False,
) -> LeakageCheckResult:
    """Verify that no leakage group or exact SHA-256 hash crosses split boundaries.

    Also flags contradictory REAL/FAKE labels within the same leakage group.
    """
    if df.empty:
        return LeakageCheckResult(
            passed=True,
            split_column=split_column,
            cross_split_group_count=0,
            cross_label_group_count=0,
            cross_split_groups=[],
            cross_label_groups=[],
        )

    if split_column not in df.columns or group_column not in df.columns:
        raise ValueError(
            f"Leakage check requires columns '{split_column}' and '{group_column}' in DataFrame."
        )

    groups_splits: dict[str, set[str]] = defaultdict(set)
    groups_labels: dict[str, set[int]] = defaultdict(set)
    groups_ids: dict[str, list[str]] = defaultdict(list)

    for img_id, grp, sp, lbl in zip(
        df["image_id"].astype(str).tolist(),
        df[group_column].astype(str).tolist(),
        df[split_column].astype(str).tolist(),
        df["label"].astype(int).tolist(),
        strict=True,
    ):
        groups_splits[grp].add(sp)
        groups_labels[grp].add(lbl)
        groups_ids[grp].append(img_id)

    # Also enforce exact SHA-256 check independently in case new hashes were materialized
    if sha256_column in df.columns:
        for img_id, sha, sp, lbl in zip(
            df["image_id"].astype(str).tolist(),
            df[sha256_column].tolist(),
            df[split_column].astype(str).tolist(),
            df["label"].astype(int).tolist(),
            strict=True,
        ):
            if sha and isinstance(sha, str) and sha.strip():
                sha_key = f"sha256:{sha.strip()}"
                groups_splits[sha_key].add(sp)
                groups_labels[sha_key].add(lbl)
                groups_ids[sha_key].append(img_id)

    cross_split: list[dict[str, Any]] = []
    cross_label: list[dict[str, Any]] = []

    for grp in sorted(groups_splits.keys()):
        if len(groups_splits[grp]) > 1:
            cross_split.append(
                {
                    "group": grp,
                    "splits": sorted(groups_splits[grp]),
                    "image_ids": sorted(set(groups_ids[grp]))[:10],
                    "count": len(set(groups_ids[grp])),
                }
            )
        if len(groups_labels[grp]) > 1:
            cross_label.append(
                {
                    "group": grp,
                    "labels": sorted(groups_labels[grp]),
                    "image_ids": sorted(set(groups_ids[grp]))[:10],
                    "count": len(set(groups_ids[grp])),
                }
            )

    passed = len(cross_split) == 0 and len(cross_label) == 0
    result = LeakageCheckResult(
        passed=passed,
        split_column=split_column,
        cross_split_group_count=len(cross_split),
        cross_label_group_count=len(cross_label),
        cross_split_groups=cross_split[:50],
        cross_label_groups=cross_label[:50],
    )

    if raise_on_violation and not passed:
        raise LeakageViolationError(
            f"Leakage violation detected on '{split_column}': "
            f"{result.cross_split_group_count} cross-split groups, "
            f"{result.cross_label_group_count} cross-label groups. "
            f"Sample: {(cross_split or cross_label)[:3]}"
        )

    return result


def filter_cross_source_split_leakage(
    df: pd.DataFrame,
    policy: str = "fail",
) -> tuple[pd.DataFrame, LeakageCheckResult]:
    """Check upstream `source_split` leakage and either fail or quarantine leaking groups.

    Never silently moves images between upstream train/val splits.
    """
    check = check_leakage_boundaries(df, split_column="source_split", raise_on_violation=False)
    if check.passed:
        return df, check

    if policy == "fail":
        raise LeakageViolationError(
            f"Upstream source_split leakage detected ({check.cross_split_group_count} cross-split "
            f"groups, {check.cross_label_group_count} cross-label groups) under policy='fail'. "
            "Either resolve upstream duplicates or set cross_source_split_leakage_policy='exclude_cross_split_groups'."
        )

    if policy == "exclude_cross_split_groups":
        bad_groups = {
            g["group"]
            for g in (check.cross_split_groups + check.cross_label_groups)
            if not g["group"].startswith("sha256:")
        }
        bad_shas = {
            g["group"].removeprefix("sha256:")
            for g in (check.cross_split_groups + check.cross_label_groups)
            if g["group"].startswith("sha256:")
        }
        out = df.copy()
        mask = out["leakage_group_id"].isin(bad_groups)
        if bad_shas and "sha256" in out.columns:
            mask = mask | out["sha256"].isin(bad_shas)
        out.loc[mask, "eligibility_status"] = "EXCLUDED"
        out.loc[mask, "exclusion_reason"] = "CROSS_SOURCE_SPLIT_OR_LABEL_LEAKAGE"
        return out, check

    raise ValueError(f"Unsupported cross_source_split_leakage_policy: {policy!r}")
