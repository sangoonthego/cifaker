"""Lightweight throughput and peak-memory profiling utility for the ProofLens data pipeline.

Measures actual wall-clock latency, throughput (items/sec and MB/sec), and Python
heap peak memory (`tracemalloc`) across:
- Inventory enumeration
- Release planning & metadata processing
- Image decoding + SHA-256 + TAR shard writing (materialization)
- Consumer cache preparation
- PyTorch TAR shard reading
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Any

from torchvision import transforms

from prooflens.data.config import ReleaseConfig
from prooflens.data.registry.inventory import build_source_inventory
from prooflens.data.registry.schema import read_canonical_parquet
from prooflens.data.releases.manager import ReleaseConsumer, ReleaseManager, load_release
from prooflens.data.shard_dataset import create_shard_dataloader
from prooflens.data.sources.genimage import create_source_adapter
from prooflens.data.storage import create_storage_backend


def _measure_stage(fn: Any) -> tuple[Any, float, float]:
    """Run `fn()` and return `(result, duration_seconds, peak_python_memory_mb)`."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        res = fn()
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return res, elapsed, peak_bytes / (1024 * 1024)


def run_pipeline_benchmark(
    config: ReleaseConfig,
    cache_dir: Path | str,
    split: str = "train",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Profile end-to-end data pipeline stages on a configured source & storage backend."""
    adapter = create_source_adapter(config.source)
    storage = create_storage_backend(config.storage)
    work_dir = Path(config.working_dir) / f"bench_{config.release_id}"
    inv_path = work_dir / "inventory.parquet"

    inv_res, inv_sec, inv_peak_mb = _measure_stage(
        lambda: build_source_inventory(
            adapter=adapter,
            output_parquet=inv_path,
            batch_size=config.inventory_batch_size,
        )
    )
    inv_df = read_canonical_parquet(inv_res.parquet_path)

    manager = ReleaseManager(storage)
    plan_res, plan_sec, plan_peak_mb = _measure_stage(
        lambda: manager.plan(inv_df, config, adapter=adapter, dry_run=False)
    )

    mat_res, mat_sec, mat_peak_mb = _measure_stage(
        lambda: manager.build(inv_df, adapter=adapter, config=config, resume=True)
    )

    ver_res, ver_sec, ver_peak_mb = _measure_stage(
        lambda: manager.verify(config.release_id, verify_tar_members=True)
    )
    if not storage.is_committed_release(config.release_id):
        manager.publish(config.release_id)

    consumer = ReleaseConsumer(storage=storage, cache_dir=cache_dir)
    prep_res, prep_sec, prep_peak_mb = _measure_stage(
        lambda: consumer.prepare_split(config.release_id, split=split, verify_checksums=True)
    )

    # Measure PyTorch shard reading throughput
    tfm = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def _read_all_batches() -> int:
        ds = load_release(
            release=config.release_id,
            split=split,
            cache_dir=cache_dir,
            storage=storage,
            transform=tfm,
        )
        loader = create_shard_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        count = 0
        for batch in loader:
            count += len(batch["image_id"])
        return count

    read_count, read_sec, read_peak_mb = _measure_stage(_read_all_batches)

    total_shard_bytes = sum(r.byte_size for r in mat_res.receipts)
    return {
        "release_id": config.release_id,
        "inventory": {
            "records": inv_res.total_records,
            "seconds": round(inv_sec, 4),
            "records_per_sec": round(inv_res.total_records / max(inv_sec, 1e-6), 2),
            "peak_python_mb": round(inv_peak_mb, 3),
        },
        "planning": {
            "selected_records": len(plan_res.selection_df),
            "seconds": round(plan_sec, 4),
            "records_per_sec": round(len(plan_res.selection_df) / max(plan_sec, 1e-6), 2),
            "peak_python_mb": round(plan_peak_mb, 3),
        },
        "materialization": {
            "images_processed": mat_res.metrics.get("images_decoded_and_hashed", 0),
            "total_shard_bytes": total_shard_bytes,
            "seconds": round(mat_sec, 4),
            "images_per_sec": round(
                mat_res.metrics.get("images_decoded_and_hashed", 0) / max(mat_sec, 1e-6), 2
            ),
            "mb_per_sec": round((total_shard_bytes / (1024 * 1024)) / max(mat_sec, 1e-6), 2),
            "peak_python_mb": round(mat_peak_mb, 3),
        },
        "verification": {
            "passed": ver_res.passed,
            "seconds": round(ver_sec, 4),
            "peak_python_mb": round(ver_peak_mb, 3),
        },
        "cache_prepare": {
            "split": split,
            "shards_staged": len(prep_res.shard_paths),
            "cache_hits": prep_res.cache_hits,
            "cache_misses": prep_res.cache_misses,
            "seconds": round(prep_sec, 4),
            "peak_python_mb": round(prep_peak_mb, 3),
        },
        "shard_dataloader_read": {
            "samples_read": read_count,
            "seconds": round(read_sec, 4),
            "samples_per_sec": round(read_count / max(read_sec, 1e-6), 2),
            "peak_python_mb": round(read_peak_mb, 3),
        },
    }
