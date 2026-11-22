"""Deterministic nested sampling, stratification, and leakage control."""

from __future__ import annotations

from prooflens.data.sampling.leakage_guard import (
    LeakageCheckResult,
    LeakageViolationError,
    check_leakage_boundaries,
    filter_cross_source_split_leakage,
)
from prooflens.data.sampling.planner import (
    ReleasePlan,
    plan_release,
    plan_replacements_for_exclusions,
    verify_plan_replay,
)
from prooflens.data.sampling.stratification import (
    QuotaInfeasibleError,
    StratumSelectionResult,
    analyze_confounding_risks,
    deterministic_group_rank_key,
    distribute_quota_evenly,
    select_groups_deterministically,
)

__all__ = [
    "LeakageCheckResult",
    "LeakageViolationError",
    "QuotaInfeasibleError",
    "ReleasePlan",
    "StratumSelectionResult",
    "analyze_confounding_risks",
    "check_leakage_boundaries",
    "deterministic_group_rank_key",
    "distribute_quota_evenly",
    "filter_cross_source_split_leakage",
    "plan_release",
    "plan_replacements_for_exclusions",
    "select_groups_deterministically",
    "verify_plan_replay",
]
