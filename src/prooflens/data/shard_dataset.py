"""TAR-aware PyTorch Dataset and DataLoader integration for ProofLens dataset releases.

Provides two production-ready readers over committed uncompressed TAR shards:
1. `ShardMapDataset(Dataset)`:
   - O(1) lazy byte-offset seeks directly into locally staged TAR shards (`shard_offset_data`, `file_size_bytes`).
   - Exact `__len__`, deterministic evaluation iteration, seeded reproducible shuffled training via
     PyTorch `DataLoader`, and multi-worker safety (per-worker lazy file handle cache).
   - Never permanently enforces 224x224 resizing or ImageNet normalization; caller passes model-specific
     `transform` (if `transform=None`, returns a raw PIL RGB Image or optional tensor conversion).
   - Fails visibly with `CorruptedShardMemberError` on truncated/corrupt bytes or SHA-256 mismatch
     (never silently skips samples during benchmark evaluation).

2. `ShardIterableDataset(IterableDataset)`:
   - Streams shard-by-shard using a `ShardCacheManager` (`FullLocalShardCache` or `BoundedShardCache`),
     enabling bounded-disk training on 500K+ image releases.
   - Explicit epoch-length semantics (`__len__` equals exact row count in `manifest_df`).
   - Deterministic multi-worker sharding: partitions shards (or row indices when shards < workers)
     disjointly across workers to prevent unintended duplicate sampling.
   - Optional deterministic shard-order shuffling + bounded in-memory shuffle buffer seeded by `(seed, epoch)`.
   - Honest limitation: mid-epoch exact resume is NOT claimed or implemented; resuming restarts at
     epoch granularity via `set_epoch(epoch)`.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from prooflens.data.releases.manager import ShardCacheManager

ImageFile.LOAD_TRUNCATED_IMAGES = False


class CorruptedShardMemberError(RuntimeError):
    """Raised when a TAR shard member is missing, truncated, corrupted, or fails SHA-256 check."""


def _decode_member_bytes(
    raw_bytes: bytes,
    image_id: str,
    shard_id: str,
    expected_sha256: str | None = None,
    verify_member_sha256: bool = False,
) -> Image.Image:
    if not raw_bytes:
        raise CorruptedShardMemberError(
            f"Empty image bytes for image_id='{image_id}' in shard='{shard_id}'."
        )
    if verify_member_sha256 and expected_sha256:
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != expected_sha256:
            raise CorruptedShardMemberError(
                f"Member SHA-256 mismatch for image_id='{image_id}' in shard='{shard_id}': "
                f"expected {expected_sha256}, got {actual_sha}."
            )
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            return img.convert("RGB")
    except Exception as exc:
        raise CorruptedShardMemberError(
            f"Failed to decode image_id='{image_id}' from shard='{shard_id}' "
            f"({type(exc).__name__}): {exc}"
        ) from exc


class ShardMapDataset(Dataset[dict[str, Any]]):
    """Random-access PyTorch Dataset reading lazily via byte offsets inside staged TAR shards."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        shard_paths: dict[str, Path],
        transform: Callable[[Image.Image], Any] | None = None,
        verify_member_sha256: bool = False,
    ) -> None:
        required = (
            "image_id",
            "label",
            "generator",
            "source_generator_group",
            "source_split",
            "release_split",
            "shard_id",
            "shard_offset_data",
            "file_size_bytes",
        )
        missing = [c for c in required if c not in manifest_df.columns]
        if missing:
            raise ValueError(f"ShardMapDataset manifest missing required columns: {missing}")

        self.manifest = manifest_df.reset_index(drop=True)
        self.shard_paths = {str(k): Path(v) for k, v in shard_paths.items()}
        self.transform = transform
        self.verify_member_sha256 = verify_member_sha256

        # Pre-extract lightweight Python tuples for fast __getitem__ without Pandas overhead
        self._records: list[tuple[Any, ...]] = list(
            zip(
                self.manifest["image_id"].astype(str).tolist(),
                self.manifest["label"].astype(int).tolist(),
                self.manifest["generator"].astype(str).tolist(),
                self.manifest["source_generator_group"].astype(str).tolist(),
                self.manifest["source_split"].astype(str).tolist(),
                self.manifest["release_split"].astype(str).tolist(),
                self.manifest["shard_id"].astype(str).tolist(),
                self.manifest["shard_offset_data"].astype(int).tolist(),
                self.manifest["file_size_bytes"].astype(int).tolist(),
                (
                    self.manifest["sha256"].tolist()
                    if "sha256" in self.manifest.columns
                    else [None] * len(self.manifest)
                ),
                strict=True,
            )
        )

        # Lazily opened per-process/per-worker file handles (never pickled across workers)
        self._handles: dict[str, BinaryIO] = {}
        self._handles_pid: int | None = None

    def __len__(self) -> int:
        return len(self._records)

    def _get_handle(self, shard_id: str) -> BinaryIO:
        pid = os.getpid()
        if self._handles_pid != pid:
            self.close()
            self._handles_pid = pid
        if shard_id not in self._handles:
            path = self.shard_paths.get(shard_id)
            if path is None or not path.is_file():
                raise CorruptedShardMemberError(
                    f"Staged TAR shard '{shard_id}' is missing at {path}."
                )
            self._handles[shard_id] = path.open("rb")
        return self._handles[shard_id]

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handles_pid"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        (
            image_id,
            label,
            generator,
            gen_group,
            source_split,
            release_split,
            shard_id,
            offset_data,
            byte_size,
            expected_sha,
        ) = self._records[index]

        handle = self._get_handle(shard_id)
        handle.seek(offset_data)
        raw_bytes = handle.read(byte_size)
        if len(raw_bytes) != byte_size:
            raise CorruptedShardMemberError(
                f"Truncated read for image_id='{image_id}' in shard='{shard_id}': "
                f"expected {byte_size} bytes at offset {offset_data}, got {len(raw_bytes)}."
            )

        rgb_img = _decode_member_bytes(
            raw_bytes=raw_bytes,
            image_id=image_id,
            shard_id=shard_id,
            expected_sha256=expected_sha,
            verify_member_sha256=self.verify_member_sha256,
        )
        image_out = self.transform(rgb_img) if self.transform is not None else rgb_img

        return {
            "image": image_out,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_id": image_id,
            "generator": generator,
            "source_generator_group": gen_group,
            "source_split": source_split,
            "release_split": release_split,
            "shard_id": shard_id,
        }


class ShardIterableDataset(IterableDataset[dict[str, Any]]):
    """Streaming TAR shard IterableDataset with explicit epoch length and worker sharding.

    Behavior:
    - Epoch length (`len(dataset)`) equals the exact number of rows in `manifest_df`.
    - Worker sharding: assigns disjoint shards (or disjoint row slices when `num_shards < num_workers`)
      so multi-worker `DataLoader` yields every sample in the split **exactly once per epoch**.
    - Shuffling: when `shuffle_shards=True`, shuffles shard order and applies a bounded
      `shuffle_buffer_size` deterministically seeded by `(seed, epoch)`.
    - Mid-epoch resume is NOT supported; call `set_epoch(e)` at epoch boundaries.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        shards_metadata: dict[str, dict[str, Any]],
        shard_cache: ShardCacheManager,
        transform: Callable[[Image.Image], Any] | None = None,
        seed: int = 42,
        shuffle_shards: bool = False,
        shuffle_buffer_size: int = 0,
        verify_cached_checksum: bool = True,
        verify_member_sha256: bool = False,
    ) -> None:
        super().__init__()
        self.manifest = manifest_df.reset_index(drop=True)
        self.shards_metadata = shards_metadata
        self.shard_cache = shard_cache
        self.transform = transform
        self.seed = int(seed)
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.verify_cached_checksum = verify_cached_checksum
        self.verify_member_sha256 = verify_member_sha256
        self._epoch = 0

        # Group row tuples per shard in deterministic order
        self._shard_rows: dict[str, list[tuple[Any, ...]]] = {}
        for shard_id, grp in self.manifest.groupby("shard_id", sort=True):
            ordered = grp.sort_values(by=["shard_offset_data", "image_id"], kind="mergesort")
            self._shard_rows[str(shard_id)] = list(
                zip(
                    ordered["image_id"].astype(str).tolist(),
                    ordered["label"].astype(int).tolist(),
                    ordered["generator"].astype(str).tolist(),
                    ordered["source_generator_group"].astype(str).tolist(),
                    ordered["source_split"].astype(str).tolist(),
                    ordered["release_split"].astype(str).tolist(),
                    ordered["shard_id"].astype(str).tolist(),
                    ordered["shard_offset_data"].astype(int).tolist(),
                    ordered["file_size_bytes"].astype(int).tolist(),
                    (
                        ordered["sha256"].tolist()
                        if "sha256" in ordered.columns
                        else [None] * len(ordered)
                    ),
                    strict=True,
                )
            )
        self._shard_ids = sorted(self._shard_rows.keys())

    def set_epoch(self, epoch: int) -> None:
        """Advance epoch counter for reproducible epoch-level shard shuffling."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.manifest)

    def _iter_records_for_worker(self) -> list[tuple[str, list[tuple[Any, ...]]]]:
        rng = random.Random(self.seed + self._epoch)
        shard_order = list(self._shard_ids)
        if self.shuffle_shards:
            rng.shuffle(shard_order)

        worker = get_worker_info()
        if worker is None or worker.num_workers <= 1:
            return [(sid, self._shard_rows[sid]) for sid in shard_order]

        wid = worker.id
        n_workers = worker.num_workers
        if len(shard_order) >= n_workers:
            # Disjoint shard assignment across workers
            my_shards = shard_order[wid::n_workers]
            return [(sid, self._shard_rows[sid]) for sid in my_shards]

        # Fewer shards than workers: partition rows disjointly within each shard
        work_plan: list[tuple[str, list[tuple[Any, ...]]]] = []
        for sid in shard_order:
            rows = self._shard_rows[sid]
            my_rows = rows[wid::n_workers]
            if my_rows:
                work_plan.append((sid, my_rows))
        return work_plan

    def _raw_sample_stream(self) -> Iterator[dict[str, Any]]:
        for shard_id, rows in self._iter_records_for_worker():
            s_info = self.shards_metadata[shard_id]
            local_tar = self.shard_cache.ensure_shard(
                storage_key=str(s_info["storage_key"]),
                expected_sha256=str(s_info["sha256"]),
                expected_size=int(s_info["byte_size"]),
                verify_cached_checksum=self.verify_cached_checksum,
            )
            with local_tar.open("rb") as handle:
                for (
                    image_id,
                    label,
                    generator,
                    gen_group,
                    source_split,
                    release_split,
                    _sid,
                    offset_data,
                    byte_size,
                    expected_sha,
                ) in rows:
                    handle.seek(offset_data)
                    raw_bytes = handle.read(byte_size)
                    if len(raw_bytes) != byte_size:
                        raise CorruptedShardMemberError(
                            f"Truncated read for image_id='{image_id}' in shard='{shard_id}'."
                        )
                    rgb_img = _decode_member_bytes(
                        raw_bytes=raw_bytes,
                        image_id=image_id,
                        shard_id=shard_id,
                        expected_sha256=expected_sha,
                        verify_member_sha256=self.verify_member_sha256,
                    )
                    image_out = self.transform(rgb_img) if self.transform is not None else rgb_img
                    yield {
                        "image": image_out,
                        "label": torch.tensor(label, dtype=torch.float32),
                        "image_id": image_id,
                        "generator": generator,
                        "source_generator_group": gen_group,
                        "source_split": source_split,
                        "release_split": release_split,
                        "shard_id": shard_id,
                    }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        stream = self._raw_sample_stream()
        if not self.shuffle_shards or self.shuffle_buffer_size <= 1:
            yield from stream
            return

        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + self._epoch * 9973 + wid)
        buf: list[dict[str, Any]] = []
        for sample in stream:
            if len(buf) < self.shuffle_buffer_size:
                buf.append(sample)
            else:
                idx = rng.randrange(len(buf))
                yield buf[idx]
                buf[idx] = sample
        rng.shuffle(buf)
        yield from buf


def create_shard_dataloader(
    dataset: ShardMapDataset | ShardIterableDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Construct a reproducible PyTorch DataLoader over either `ShardMapDataset` or `ShardIterableDataset`."""
    is_iterable = isinstance(dataset, IterableDataset)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if is_iterable else shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=(None if is_iterable else generator),
    )
