"""Explicit release lifecycle state machine and durable state tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from prooflens.data.storage.base import ImmutabilityError, StorageBackend


class ReleaseState(StrEnum):
    """Explicit dataset release lifecycle states."""

    PLANNED = "PLANNED"
    MATERIALIZING = "MATERIALIZING"
    MATERIALIZED = "MATERIALIZED"
    VERIFIED = "VERIFIED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass
class ReleaseStateRecord:
    """Persisted state snapshot for an in-progress or committed release."""

    release_id: str
    state: str
    updated_at_utc: str
    plan_id: str | None = None
    config_hash: str | None = None
    parent_release_id: str | None = None
    committed_shards: int = 0
    total_images: int = 0
    excluded_images: int = 0
    last_error: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReleaseStateRecord:
        return cls(
            release_id=str(data["release_id"]),
            state=str(data["state"]),
            updated_at_utc=str(data["updated_at_utc"]),
            plan_id=data.get("plan_id"),
            config_hash=data.get("config_hash"),
            parent_release_id=data.get("parent_release_id"),
            committed_shards=int(data.get("committed_shards", 0)),
            total_images=int(data.get("total_images", 0)),
            excluded_images=int(data.get("excluded_images", 0)),
            last_error=data.get("last_error"),
            history=list(data.get("history", [])),
        )


class ReleaseStateTracker:
    """Persists and reads `releases/{release_id}/state.json` on a StorageBackend."""

    def __init__(self, storage: StorageBackend, release_id: str) -> None:
        self._storage = storage
        self._release_id = release_id
        self._key = f"releases/{release_id}/state.json"

    @property
    def state_key(self) -> str:
        return self._key

    def load(self) -> ReleaseStateRecord | None:
        if not self._storage.exists(self._key):
            return None
        raw = json.loads(self._storage.read_bytes(self._key).decode("utf-8"))
        return ReleaseStateRecord.from_dict(raw)

    def transition(
        self,
        new_state: ReleaseState,
        plan_id: str | None = None,
        config_hash: str | None = None,
        parent_release_id: str | None = None,
        committed_shards: int | None = None,
        total_images: int | None = None,
        excluded_images: int | None = None,
        error_message: str | None = None,
        note: str = "",
    ) -> ReleaseStateRecord:
        if self._storage.is_committed_release(self._release_id):
            raise ImmutabilityError(
                f"Release '{self._release_id}' is already PUBLISHED; state cannot be mutated."
            )

        now = datetime.now(UTC).isoformat()
        existing = self.load()
        history = list(existing.history) if existing is not None else []
        history.append({"state": new_state.value, "timestamp_utc": now, "note": note})

        record = ReleaseStateRecord(
            release_id=self._release_id,
            state=new_state.value,
            updated_at_utc=now,
            plan_id=plan_id if plan_id is not None else (existing.plan_id if existing else None),
            config_hash=(
                config_hash
                if config_hash is not None
                else (existing.config_hash if existing else None)
            ),
            parent_release_id=(
                parent_release_id
                if parent_release_id is not None
                else (existing.parent_release_id if existing else None)
            ),
            committed_shards=(
                committed_shards
                if committed_shards is not None
                else (existing.committed_shards if existing else 0)
            ),
            total_images=(
                total_images
                if total_images is not None
                else (existing.total_images if existing else 0)
            ),
            excluded_images=(
                excluded_images
                if excluded_images is not None
                else (existing.excluded_images if existing else 0)
            ),
            last_error=error_message,
            history=history,
        )
        self._storage.write_bytes(
            json.dumps(record.to_dict(), indent=2).encode("utf-8"),
            self._key,
            overwrite=True,
        )
        return record
