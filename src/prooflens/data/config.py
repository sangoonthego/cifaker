"""Validated YAML configuration schemas for the ProofLens scalable data pipeline.

Organizes configuration by responsibility:
- SourceConfig   (`configs/sources/*.yaml`)
- ProtocolConfig (`configs/protocols/*.yaml`)
- StorageConfig  (`configs/storage/*.yaml`)
- ReleaseConfig  (`configs/releases/*.yaml`)

Every configuration validates required keys, rejects unknown critical fields,
and produces a canonical deterministic SHA-256 `config_hash` recorded in release metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_SOURCE_TYPES: frozenset[str] = frozenset(
    {"directory", "archive", "huggingface", "huggingface_arrow"}
)
VALID_STORAGE_BACKENDS: frozenset[str] = frozenset({"local", "google_drive"})
VALID_GROUP_QUOTA_POLICIES: frozenset[str] = frozenset(
    {"exact_or_fail", "skip_oversized_group", "allow_underfill"}
)
VALID_CROSS_SPLIT_LEAKAGE_POLICIES: frozenset[str] = frozenset(
    {"fail", "exclude_cross_split_groups"}
)
VALID_GDRIVE_MODES: frozenset[str] = frozenset({"api", "mounted"})
VALID_NEAR_DUP_GATES: frozenset[str] = frozenset({"warn", "block_publish"})


class ConfigValidationError(ValueError):
    """Raised when a pipeline YAML configuration fails schema validation."""


def _reject_unknown_keys(
    mapping: dict[str, Any], allowed: set[str] | frozenset[str], context: str
) -> None:
    unknown = sorted(set(mapping.keys()) - set(allowed))
    if unknown:
        raise ConfigValidationError(
            f"Unknown configuration field(s) in {context}: {unknown}. Allowed: {sorted(allowed)}"
        )


def _load_yaml_dict(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigValidationError(f"Expected YAML mapping in {p}, got {type(data).__name__}")
    return data


def _resolve_ref_path(ref_value: str | Path, base_dir: Path) -> Path:
    ref_path = Path(ref_value)
    if ref_path.is_absolute() or ref_path.is_file():
        return ref_path
    candidate = base_dir / ref_path
    if candidate.is_file():
        return candidate
    return ref_path


def compute_config_hash(payload: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a resolved configuration dictionary."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceConfig:
    """Validated upstream dataset source configuration."""

    source_id: str
    source_type: str
    source_revision: str
    location: str
    label_mapping: dict[str, int] = field(
        default_factory=lambda: {"nature": 0, "real": 0, "ai": 1, "fake": 1}
    )
    split_mapping: dict[str, str] = field(
        default_factory=lambda: {"train": "train", "val": "val", "valid": "val", "test": "test"}
    )
    generator_mapping: dict[str, str] = field(default_factory=dict)
    strict_generator_mapping: bool = False
    real_generator_value: str = "none"
    supported_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    )
    archive_glob: str | None = None
    hf_config_name: str | None = None
    hf_column_mapping: dict[str, str] = field(default_factory=dict)
    hf_streaming: bool = True
    estimated_total_bytes: int | None = None
    arrow_cache_dir: str | None = None
    arrow_max_cached_shards: int = 2
    arrow_max_cache_bytes: int | None = None
    arrow_batch_size: int = 256
    arrow_download_retries: int = 3
    arrow_retry_backoff_seconds: float = 1.0
    hf_token_env_var: str = "HF_TOKEN"
    arrow_local_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitRuleConfig:
    """Sampling constraints for a single target split in a protocol."""

    allowed_source_splits: tuple[str, ...]
    generator_pool: str
    frozen_across_scaling: bool
    real_quota: int
    fake_quota: int
    stratify_by_generator_group: bool = True
    stratify_by_imagenet_class: bool = False

    @property
    def total_quota(self) -> int:
        return self.real_quota + self.fake_quota

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolConfig:
    """Validated cross-generator experimental dataset protocol."""

    protocol_version: str
    seen_generators: tuple[str, ...]
    unseen_generators: tuple[str, ...]
    splits: dict[str, SplitRuleConfig]
    group_quota_policy: str = "exact_or_fail"
    cross_source_split_leakage_policy: str = "fail"
    deduplicate_within_group: bool = True
    include_perceptual_in_leakage_groups: bool = False
    perceptual_hamming_threshold: int = 4

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["splits"] = {k: v.to_dict() for k, v in self.splits.items()}
        return out


@dataclass(frozen=True)
class StorageConfig:
    """Validated storage backend configuration (local filesystem or Google Drive)."""

    backend: str
    root_path: str | None = None
    gdrive_folder_id: str | None = None
    gdrive_folder_name: str = "cifaker"
    gdrive_mode: str = "api"
    gdrive_mount_root: str | None = None
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    verify_after_upload: bool = True
    single_writer_lock: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseConfig:
    """Top-level validated configuration for building or verifying a dataset release."""

    release_id: str
    sampling_seed: int
    source: SourceConfig
    protocol: ProtocolConfig
    storage: StorageConfig
    parent_release_id: str | None = None
    shard_target_bytes: int = 1024 * 1024 * 1024
    max_samples_per_shard: int | None = None
    materialization_workers: int = 4
    materialization_batch_size: int = 256
    source_max_retries: int = 3
    compute_pixel_hash: bool = False
    compute_perceptual_hash: bool = True
    inventory_batch_size: int = 5000
    working_dir: str = "artifacts/pipeline_work"
    description: str = ""
    frozen_snapshot_prefix: str | None = None
    strict_frozen_mode: bool = False
    near_duplicate_quality_gate: str = "warn"

    def to_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "release_id": self.release_id,
            "sampling_seed": self.sampling_seed,
            "parent_release_id": self.parent_release_id,
            "shard_target_bytes": self.shard_target_bytes,
            "max_samples_per_shard": self.max_samples_per_shard,
            "materialization_workers": self.materialization_workers,
            "materialization_batch_size": self.materialization_batch_size,
            "source_max_retries": self.source_max_retries,
            "compute_pixel_hash": self.compute_pixel_hash,
            "compute_perceptual_hash": self.compute_perceptual_hash,
            "inventory_batch_size": self.inventory_batch_size,
            "working_dir": self.working_dir,
            "description": self.description,
            "source": self.source.to_dict(),
            "protocol": self.protocol.to_dict(),
            "storage": self.storage.to_dict(),
        }
        if self.frozen_snapshot_prefix is not None:
            base["frozen_snapshot_prefix"] = self.frozen_snapshot_prefix
        if self.strict_frozen_mode:
            base["strict_frozen_mode"] = self.strict_frozen_mode
        if self.near_duplicate_quality_gate != "warn":
            base["near_duplicate_quality_gate"] = self.near_duplicate_quality_gate
        return base

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self.to_dict())


def validate_source_config(raw: dict[str, Any]) -> SourceConfig:
    """Validate a source configuration dictionary."""
    body = raw["source"] if "source" in raw and isinstance(raw["source"], dict) else raw
    allowed = {
        "source_id",
        "source_type",
        "source_revision",
        "location",
        "label_mapping",
        "split_mapping",
        "generator_mapping",
        "strict_generator_mapping",
        "real_generator_value",
        "supported_extensions",
        "archive_glob",
        "hf_config_name",
        "hf_column_mapping",
        "hf_streaming",
        "estimated_total_bytes",
        "arrow_cache_dir",
        "arrow_max_cached_shards",
        "arrow_max_cache_bytes",
        "arrow_batch_size",
        "arrow_download_retries",
        "arrow_retry_backoff_seconds",
        "hf_token_env_var",
        "arrow_local_root",
    }
    _reject_unknown_keys(body, allowed, "source")

    source_id = str(body.get("source_id", "")).strip()
    source_type = str(body.get("source_type", "")).strip().lower()
    source_revision = str(body.get("source_revision", "")).strip()
    location = str(body.get("location", "")).strip()

    if not source_id:
        raise ConfigValidationError("Missing required field 'source.source_id'.")
    if source_type not in VALID_SOURCE_TYPES:
        raise ConfigValidationError(
            f"Unsupported 'source.source_type': '{source_type}'. Expected one of {sorted(VALID_SOURCE_TYPES)}."
        )
    if not source_revision:
        raise ConfigValidationError("Missing required field 'source.source_revision'.")
    if not location:
        raise ConfigValidationError("Missing required field 'source.location'.")

    raw_label_map = body.get("label_mapping", {"nature": 0, "real": 0, "ai": 1, "fake": 1})
    if not isinstance(raw_label_map, dict) or not raw_label_map:
        raise ConfigValidationError("'source.label_mapping' must be a non-empty mapping.")
    label_mapping: dict[str, int] = {}
    for k, v in raw_label_map.items():
        if v not in (0, 1):
            raise ConfigValidationError(
                f"Invalid binary label {v!r} for key {k!r} in 'source.label_mapping'."
            )
        label_mapping[str(k).lower()] = int(v)

    raw_split_map = body.get(
        "split_mapping", {"train": "train", "val": "val", "valid": "val", "test": "test"}
    )
    if not isinstance(raw_split_map, dict) or not raw_split_map:
        raise ConfigValidationError("'source.split_mapping' must be a non-empty mapping.")
    split_mapping = {str(k).lower(): str(v).strip() for k, v in raw_split_map.items()}

    raw_gen_map = body.get("generator_mapping", {}) or {}
    if not isinstance(raw_gen_map, dict):
        raise ConfigValidationError("'source.generator_mapping' must be a mapping.")
    generator_mapping = {str(k): str(v).strip() for k, v in raw_gen_map.items()}

    real_gen_val = str(body.get("real_generator_value", "none")).strip().lower()
    if real_gen_val not in {"none", "real", "natural"}:
        raise ConfigValidationError(
            "'source.real_generator_value' must be one of {'none', 'real', 'natural'} "
            "to avoid falsely attributing REAL images to a generative model."
        )

    exts = tuple(
        str(e).lower()
        for e in body.get(
            "supported_extensions", (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")
        )
    )
    if not exts:
        raise ConfigValidationError("'source.supported_extensions' cannot be empty.")

    est_bytes = body.get("estimated_total_bytes")
    if est_bytes is not None:
        est_bytes = int(est_bytes)
        if est_bytes < 0:
            raise ConfigValidationError("'source.estimated_total_bytes' must be non-negative.")

    hf_col_map = body.get("hf_column_mapping", {}) or {}
    if not isinstance(hf_col_map, dict):
        raise ConfigValidationError("'source.hf_column_mapping' must be a mapping.")

    arrow_max_shards = int(body.get("arrow_max_cached_shards", 2))
    if arrow_max_shards < 1:
        raise ConfigValidationError("'source.arrow_max_cached_shards' must be >= 1.")

    arrow_max_bytes_raw = body.get("arrow_max_cache_bytes")
    arrow_max_bytes = int(arrow_max_bytes_raw) if arrow_max_bytes_raw is not None else None
    if arrow_max_bytes is not None and arrow_max_bytes <= 0:
        raise ConfigValidationError("'source.arrow_max_cache_bytes' must be positive.")

    arrow_batch = int(body.get("arrow_batch_size", 256))
    if arrow_batch < 1:
        raise ConfigValidationError("'source.arrow_batch_size' must be >= 1.")

    arrow_retries = int(body.get("arrow_download_retries", 3))
    if arrow_retries < 1:
        raise ConfigValidationError("'source.arrow_download_retries' must be >= 1.")

    arrow_backoff = float(body.get("arrow_retry_backoff_seconds", 1.0))
    if arrow_backoff < 0.0:
        raise ConfigValidationError("'source.arrow_retry_backoff_seconds' must be >= 0.")

    return SourceConfig(
        source_id=source_id,
        source_type=source_type,
        source_revision=source_revision,
        location=location,
        label_mapping=label_mapping,
        split_mapping=split_mapping,
        generator_mapping=generator_mapping,
        strict_generator_mapping=bool(body.get("strict_generator_mapping", False)),
        real_generator_value=real_gen_val,
        supported_extensions=exts,
        archive_glob=str(body["archive_glob"]) if body.get("archive_glob") is not None else None,
        hf_config_name=(
            str(body["hf_config_name"]) if body.get("hf_config_name") is not None else None
        ),
        hf_column_mapping={str(k): str(v) for k, v in hf_col_map.items()},
        hf_streaming=bool(body.get("hf_streaming", True)),
        estimated_total_bytes=est_bytes,
        arrow_cache_dir=(
            str(body["arrow_cache_dir"]).strip()
            if body.get("arrow_cache_dir") is not None
            else None
        ),
        arrow_max_cached_shards=arrow_max_shards,
        arrow_max_cache_bytes=arrow_max_bytes,
        arrow_batch_size=arrow_batch,
        arrow_download_retries=arrow_retries,
        arrow_retry_backoff_seconds=arrow_backoff,
        hf_token_env_var=str(body.get("hf_token_env_var", "HF_TOKEN")).strip() or "HF_TOKEN",
        arrow_local_root=(
            str(body["arrow_local_root"]).strip()
            if body.get("arrow_local_root") is not None
            else None
        ),
    )


def validate_protocol_config(
    raw: dict[str, Any],
    quota_overrides: dict[str, dict[str, int]] | None = None,
) -> ProtocolConfig:
    """Validate a protocol configuration dictionary with optional release-level quota overrides."""
    body = raw["protocol"] if "protocol" in raw and isinstance(raw["protocol"], dict) else raw
    allowed = {
        "protocol_version",
        "seen_generators",
        "unseen_generators",
        "splits",
        "group_quota_policy",
        "cross_source_split_leakage_policy",
        "deduplicate_within_group",
        "include_perceptual_in_leakage_groups",
        "perceptual_hamming_threshold",
    }
    _reject_unknown_keys(body, allowed, "protocol")

    protocol_version = str(body.get("protocol_version", "")).strip()
    if not protocol_version:
        raise ConfigValidationError("Missing required field 'protocol.protocol_version'.")

    seen_raw = body.get("seen_generators")
    unseen_raw = body.get("unseen_generators")
    if not isinstance(seen_raw, list) or not seen_raw:
        raise ConfigValidationError("'protocol.seen_generators' must be a non-empty list.")
    if not isinstance(unseen_raw, list) or not unseen_raw:
        raise ConfigValidationError("'protocol.unseen_generators' must be a non-empty list.")

    seen = tuple(str(g).strip() for g in seen_raw)
    unseen = tuple(str(g).strip() for g in unseen_raw)
    overlap = set(seen) & set(unseen)
    if overlap:
        raise ConfigValidationError(
            f"Generators cannot be in both seen_generators and unseen_generators: {sorted(overlap)}"
        )

    group_quota_policy = str(body.get("group_quota_policy", "exact_or_fail")).strip()
    if group_quota_policy not in VALID_GROUP_QUOTA_POLICIES:
        raise ConfigValidationError(
            f"Invalid 'protocol.group_quota_policy': '{group_quota_policy}'. "
            f"Expected one of {sorted(VALID_GROUP_QUOTA_POLICIES)}."
        )

    leakage_policy = str(body.get("cross_source_split_leakage_policy", "fail")).strip()
    if leakage_policy not in VALID_CROSS_SPLIT_LEAKAGE_POLICIES:
        raise ConfigValidationError(
            f"Invalid 'protocol.cross_source_split_leakage_policy': '{leakage_policy}'. "
            f"Expected one of {sorted(VALID_CROSS_SPLIT_LEAKAGE_POLICIES)}."
        )

    raw_splits = body.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ConfigValidationError("'protocol.splits' must be a non-empty mapping.")

    split_allowed = {
        "allowed_source_splits",
        "generator_pool",
        "frozen_across_scaling",
        "real_quota",
        "fake_quota",
        "stratify_by_generator_group",
        "stratify_by_imagenet_class",
    }
    splits: dict[str, SplitRuleConfig] = {}
    for split_name, spec in raw_splits.items():
        if not isinstance(spec, dict):
            raise ConfigValidationError(f"Split rule '{split_name}' must be a mapping.")
        _reject_unknown_keys(spec, split_allowed, f"protocol.splits.{split_name}")

        allowed_src = spec.get("allowed_source_splits")
        if not isinstance(allowed_src, list) or not allowed_src:
            raise ConfigValidationError(
                f"Split '{split_name}' requires non-empty 'allowed_source_splits'."
            )

        gen_pool = str(spec.get("generator_pool", "")).strip().lower()
        if gen_pool not in {"seen", "unseen"}:
            raise ConfigValidationError(
                f"Split '{split_name}' 'generator_pool' must be 'seen' or 'unseen', got '{gen_pool}'."
            )

        # Enforce unseen exclusion from train and dev
        if split_name in {"train", "dev"} and gen_pool != "seen":
            raise ConfigValidationError(
                f"Split '{split_name}' must draw exclusively from 'seen' generators, got '{gen_pool}'."
            )

        override = (quota_overrides or {}).get(split_name, {})
        real_q = override.get("real_quota", spec.get("real_quota"))
        fake_q = override.get("fake_quota", spec.get("fake_quota"))
        if (
            not isinstance(real_q, int)
            or isinstance(real_q, bool)
            or real_q < 0
            or not isinstance(fake_q, int)
            or isinstance(fake_q, bool)
            or fake_q < 0
        ):
            raise ConfigValidationError(
                f"Split '{split_name}' requires non-negative integer 'real_quota' and 'fake_quota'."
            )

        splits[str(split_name)] = SplitRuleConfig(
            allowed_source_splits=tuple(str(s).strip() for s in allowed_src),
            generator_pool=gen_pool,
            frozen_across_scaling=bool(spec.get("frozen_across_scaling", split_name != "train")),
            real_quota=real_q,
            fake_quota=fake_q,
            stratify_by_generator_group=bool(spec.get("stratify_by_generator_group", True)),
            stratify_by_imagenet_class=bool(spec.get("stratify_by_imagenet_class", False)),
        )

    if quota_overrides:
        unknown_override_splits = sorted(set(quota_overrides.keys()) - set(splits.keys()))
        if unknown_override_splits:
            raise ConfigValidationError(
                f"Release quota overrides reference unknown split(s): {unknown_override_splits}"
            )

    hamming_thresh = int(body.get("perceptual_hamming_threshold", 4))
    if hamming_thresh < 0 or hamming_thresh > 64:
        raise ConfigValidationError(
            f"'protocol.perceptual_hamming_threshold' must be in [0, 64], got {hamming_thresh}."
        )

    return ProtocolConfig(
        protocol_version=protocol_version,
        seen_generators=seen,
        unseen_generators=unseen,
        splits=splits,
        group_quota_policy=group_quota_policy,
        cross_source_split_leakage_policy=leakage_policy,
        deduplicate_within_group=bool(body.get("deduplicate_within_group", True)),
        include_perceptual_in_leakage_groups=bool(
            body.get("include_perceptual_in_leakage_groups", False)
        ),
        perceptual_hamming_threshold=hamming_thresh,
    )


def validate_storage_config(raw: dict[str, Any]) -> StorageConfig:
    """Validate a storage backend configuration dictionary."""
    body = raw["storage"] if "storage" in raw and isinstance(raw["storage"], dict) else raw
    allowed = {
        "backend",
        "root_path",
        "gdrive_folder_id",
        "gdrive_folder_name",
        "gdrive_mode",
        "gdrive_mount_root",
        "max_retries",
        "retry_backoff_seconds",
        "verify_after_upload",
        "single_writer_lock",
    }
    _reject_unknown_keys(body, allowed, "storage")

    backend = str(body.get("backend", "")).strip().lower()
    if backend not in VALID_STORAGE_BACKENDS:
        raise ConfigValidationError(
            f"Unsupported 'storage.backend': '{backend}'. Expected one of {sorted(VALID_STORAGE_BACKENDS)}."
        )

    root_path = str(body["root_path"]).strip() if body.get("root_path") is not None else None
    gdrive_folder_id = (
        str(body["gdrive_folder_id"]).strip() if body.get("gdrive_folder_id") is not None else None
    )
    gdrive_mode = str(body.get("gdrive_mode", "api")).strip().lower()
    gdrive_mount_root = (
        str(body["gdrive_mount_root"]).strip()
        if body.get("gdrive_mount_root") is not None
        else None
    )

    if backend == "local" and not root_path:
        raise ConfigValidationError("Local storage backend requires 'storage.root_path'.")

    if backend == "google_drive":
        if gdrive_mode not in VALID_GDRIVE_MODES:
            raise ConfigValidationError(
                f"Invalid 'storage.gdrive_mode': '{gdrive_mode}'. Expected one of {sorted(VALID_GDRIVE_MODES)}."
            )
        if gdrive_mode == "api" and not gdrive_folder_id:
            raise ConfigValidationError(
                "Google Drive 'api' mode requires 'storage.gdrive_folder_id'."
            )
        if gdrive_mode == "mounted" and not gdrive_mount_root:
            raise ConfigValidationError(
                "Google Drive 'mounted' mode requires 'storage.gdrive_mount_root'."
            )

    max_retries = int(body.get("max_retries", 3))
    if max_retries < 1:
        raise ConfigValidationError("'storage.max_retries' must be >= 1.")

    backoff = float(body.get("retry_backoff_seconds", 1.0))
    if backoff < 0.0:
        raise ConfigValidationError("'storage.retry_backoff_seconds' must be >= 0.")

    return StorageConfig(
        backend=backend,
        root_path=root_path,
        gdrive_folder_id=gdrive_folder_id,
        gdrive_folder_name=str(body.get("gdrive_folder_name", "cifaker")).strip(),
        gdrive_mode=gdrive_mode,
        gdrive_mount_root=gdrive_mount_root,
        max_retries=max_retries,
        retry_backoff_seconds=backoff,
        verify_after_upload=bool(body.get("verify_after_upload", True)),
        single_writer_lock=bool(body.get("single_writer_lock", True)),
    )


def load_source_config(config_path: Path | str) -> SourceConfig:
    """Load and validate a standalone source YAML file."""
    return validate_source_config(_load_yaml_dict(config_path))


def load_storage_config(config_path: Path | str) -> StorageConfig:
    """Load and validate a standalone storage YAML file."""
    return validate_storage_config(_load_yaml_dict(config_path))


def load_protocol_config(config_path: Path | str) -> ProtocolConfig:
    """Load and validate a standalone protocol YAML file."""
    return validate_protocol_config(_load_yaml_dict(config_path))


def load_release_config(
    config_path: Path | str,
    storage_override: Path | str | None = None,
    source_override: Path | str | None = None,
    working_dir_override: Path | str | None = None,
) -> ReleaseConfig:
    """Load and validate a release YAML configuration, resolving referenced sub-configs."""
    path = Path(config_path)
    raw = _load_yaml_dict(path)
    base_dir = path.parent

    allowed = {
        "release_id",
        "sampling_seed",
        "parent_release_id",
        "source_config",
        "source",
        "protocol_config",
        "protocol",
        "storage_config",
        "storage",
        "split_quotas",
        "shard_target_bytes",
        "max_samples_per_shard",
        "materialization_workers",
        "materialization_batch_size",
        "source_max_retries",
        "compute_pixel_hash",
        "compute_perceptual_hash",
        "inventory_batch_size",
        "working_dir",
        "description",
        "frozen_snapshot_prefix",
        "strict_frozen_mode",
        "near_duplicate_quality_gate",
    }
    _reject_unknown_keys(raw, allowed, f"release({path.name})")

    release_id = str(raw.get("release_id", "")).strip()
    if not release_id:
        raise ConfigValidationError("Missing required field 'release_id'.")
    if "/" in release_id or "\\" in release_id or ".." in release_id:
        raise ConfigValidationError(f"Unsafe characters in 'release_id': {release_id!r}")

    seed = raw.get("sampling_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigValidationError("Field 'sampling_seed' must be an integer.")

    # Resolve source config
    if source_override is not None:
        source_cfg = load_source_config(source_override)
    elif "source" in raw and isinstance(raw["source"], dict):
        source_cfg = validate_source_config(raw["source"])
    elif "source_config" in raw:
        source_cfg = load_source_config(_resolve_ref_path(raw["source_config"], base_dir))
    else:
        raise ConfigValidationError("Release config must specify 'source_config' or 'source'.")

    # Resolve protocol config + optional split quota overrides
    split_quotas = raw.get("split_quotas")
    if split_quotas is not None and not isinstance(split_quotas, dict):
        raise ConfigValidationError("'split_quotas' must be a mapping of split -> quotas.")

    if "protocol" in raw and isinstance(raw["protocol"], dict):
        protocol_cfg = validate_protocol_config(raw["protocol"], quota_overrides=split_quotas)
    elif "protocol_config" in raw:
        proto_raw = _load_yaml_dict(_resolve_ref_path(raw["protocol_config"], base_dir))
        protocol_cfg = validate_protocol_config(proto_raw, quota_overrides=split_quotas)
    else:
        raise ConfigValidationError("Release config must specify 'protocol_config' or 'protocol'.")

    # Resolve storage config
    if storage_override is not None:
        storage_cfg = load_storage_config(storage_override)
    elif "storage" in raw and isinstance(raw["storage"], dict):
        storage_cfg = validate_storage_config(raw["storage"])
    elif "storage_config" in raw:
        storage_cfg = load_storage_config(_resolve_ref_path(raw["storage_config"], base_dir))
    else:
        raise ConfigValidationError("Release config must specify 'storage_config' or 'storage'.")

    shard_target_bytes = int(raw.get("shard_target_bytes", 1024 * 1024 * 1024))
    if shard_target_bytes <= 0:
        raise ConfigValidationError("'shard_target_bytes' must be positive.")

    max_samples_raw = raw.get("max_samples_per_shard")
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None
    if max_samples is not None and max_samples <= 0:
        raise ConfigValidationError("'max_samples_per_shard' must be positive when provided.")

    workers = int(raw.get("materialization_workers", 4))
    if workers < 1:
        raise ConfigValidationError("'materialization_workers' must be >= 1.")

    batch_size = int(raw.get("materialization_batch_size", 256))
    if batch_size < 1:
        raise ConfigValidationError("'materialization_batch_size' must be >= 1.")

    inv_batch = int(raw.get("inventory_batch_size", 5000))
    if inv_batch < 1:
        raise ConfigValidationError("'inventory_batch_size' must be >= 1.")

    parent_id = (
        str(raw["parent_release_id"]).strip() if raw.get("parent_release_id") is not None else None
    )
    if parent_id == "":
        parent_id = None

    working_dir = str(
        working_dir_override or raw.get("working_dir", "artifacts/pipeline_work")
    ).strip()

    near_dup_gate = str(raw.get("near_duplicate_quality_gate", "warn")).strip().lower()
    if near_dup_gate not in VALID_NEAR_DUP_GATES:
        raise ConfigValidationError(
            f"Invalid 'near_duplicate_quality_gate': {near_dup_gate!r}. "
            f"Expected one of {sorted(VALID_NEAR_DUP_GATES)}."
        )

    frozen_prefix = (
        str(raw["frozen_snapshot_prefix"]).strip()
        if raw.get("frozen_snapshot_prefix") is not None
        else None
    )
    if frozen_prefix == "":
        frozen_prefix = None

    return ReleaseConfig(
        release_id=release_id,
        sampling_seed=seed,
        source=source_cfg,
        protocol=protocol_cfg,
        storage=storage_cfg,
        parent_release_id=parent_id,
        shard_target_bytes=shard_target_bytes,
        max_samples_per_shard=max_samples,
        materialization_workers=workers,
        materialization_batch_size=batch_size,
        source_max_retries=int(raw.get("source_max_retries", 3)),
        compute_pixel_hash=bool(raw.get("compute_pixel_hash", False)),
        compute_perceptual_hash=bool(raw.get("compute_perceptual_hash", True)),
        inventory_batch_size=inv_batch,
        working_dir=working_dir,
        description=str(raw.get("description", "")).strip(),
        frozen_snapshot_prefix=frozen_prefix,
        strict_frozen_mode=bool(raw.get("strict_frozen_mode", False)),
        near_duplicate_quality_gate=near_dup_gate,
    )
