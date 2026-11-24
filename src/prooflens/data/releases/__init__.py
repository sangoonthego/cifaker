"""Release lifecycle state, verification, publication, and consumer APIs."""

from __future__ import annotations

from prooflens.data.releases.frozen_plan import (
    REQUIRED_FROZEN_ARTIFACTS,
    CompatibilityReport,
    FrozenPlanCompatibilityError,
    FrozenSnapshot,
    FrozenSnapshotError,
    build_materialization_provenance,
    build_pilot_selection,
    check_frozen_plan_compatibility,
    load_and_validate_frozen_snapshot,
)
from prooflens.data.releases.manager import (
    BoundedShardCache,
    FullLocalShardCache,
    PreparedReleaseSplit,
    ReleaseConsumer,
    ReleaseManager,
    ShardCacheManager,
    UnpublishedReleaseError,
    load_release,
)
from prooflens.data.releases.preflight import CapacityPreflightReport, run_capacity_preflight
from prooflens.data.releases.state import ReleaseState, ReleaseStateRecord, ReleaseStateTracker
from prooflens.data.releases.verifier import (
    ReleaseVerificationError,
    ReleaseVerificationReport,
    verify_release,
)

__all__ = [
    "BoundedShardCache",
    "CapacityPreflightReport",
    "CompatibilityReport",
    "FrozenPlanCompatibilityError",
    "FrozenSnapshot",
    "FrozenSnapshotError",
    "FullLocalShardCache",
    "PreparedReleaseSplit",
    "REQUIRED_FROZEN_ARTIFACTS",
    "ReleaseConsumer",
    "ReleaseManager",
    "ReleaseState",
    "ReleaseStateRecord",
    "ReleaseStateTracker",
    "ReleaseVerificationError",
    "ReleaseVerificationReport",
    "ShardCacheManager",
    "UnpublishedReleaseError",
    "build_materialization_provenance",
    "build_pilot_selection",
    "check_frozen_plan_compatibility",
    "load_and_validate_frozen_snapshot",
    "load_release",
    "run_capacity_preflight",
    "verify_release",
]
