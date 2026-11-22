"""Command-line interface for the ProofLens scalable dataset pipeline.

Subcommands:
- `inventory` : Enumerate an upstream source into bounded-memory canonical Parquet + summary JSON.
- `plan`      : Run deterministic nested sampling & feasibility checks (supports `--dry-run`).
- `build`     : Materialize immutable verified TAR shards with resume & parent shard reuse.
- `verify`    : Verify all shards, checksums, schemas, quotas, and leakage boundaries.
- `publish`   : Publish `release.json` last after verification passes (immutable gate).
- `prepare`   : Stage verified shards of a published release split into local cache for training.
- `benchmark` : Profile throughput and Python peak memory on a configured source & storage.

Invocation:
    python -m prooflens.data.cli <subcommand> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prooflens.data.benchmark import run_pipeline_benchmark
from prooflens.data.config import (
    load_release_config,
    load_source_config,
    load_storage_config,
)
from prooflens.data.registry.inventory import build_source_inventory
from prooflens.data.registry.schema import read_canonical_parquet
from prooflens.data.releases.manager import ReleaseConsumer, ReleaseManager
from prooflens.data.sources.genimage import create_source_adapter
from prooflens.data.storage import create_storage_backend


def _default_inventory_path(working_dir: str, source_id: str, source_revision: str) -> Path:
    safe_rev = source_revision.replace("/", "_").replace(":", "_")
    return Path(working_dir) / "inventories" / f"{source_id}_{safe_rev}.parquet"


def _cmd_inventory(args: argparse.Namespace) -> int:
    if args.source_config:
        src_cfg = load_source_config(args.source_config)
        out_parquet = (
            Path(args.output)
            if args.output
            else _default_inventory_path(
                args.working_dir or "artifacts/pipeline_work",
                src_cfg.source_id,
                src_cfg.source_revision,
            )
        )
        batch_size = args.batch_size
    else:
        rel_cfg = load_release_config(
            args.config,
            source_override=args.source_override,
            storage_override=args.storage_override,
            working_dir_override=args.working_dir,
        )
        src_cfg = rel_cfg.source
        out_parquet = (
            Path(args.output)
            if args.output
            else _default_inventory_path(
                rel_cfg.working_dir, src_cfg.source_id, src_cfg.source_revision
            )
        )
        batch_size = args.batch_size or rel_cfg.inventory_batch_size

    adapter = create_source_adapter(src_cfg)
    result = build_source_inventory(
        adapter=adapter,
        output_parquet=out_parquet,
        summary_json=Path(args.summary_out) if args.summary_out else None,
        batch_size=batch_size,
        validate_image_bytes=args.validate_bytes,
    )
    print(json.dumps(result.summary, indent=2))
    return 0


def _ensure_inventory_df(args: argparse.Namespace, rel_cfg: Any) -> Any:
    inv_path = (
        Path(args.inventory)
        if getattr(args, "inventory", None)
        else _default_inventory_path(
            rel_cfg.working_dir, rel_cfg.source.source_id, rel_cfg.source.source_revision
        )
    )
    if not inv_path.is_file():
        adapter = create_source_adapter(rel_cfg.source)
        build_source_inventory(
            adapter=adapter,
            output_parquet=inv_path,
            batch_size=rel_cfg.inventory_batch_size,
        )
    return read_canonical_parquet(inv_path)


def _cmd_plan(args: argparse.Namespace) -> int:
    rel_cfg = load_release_config(
        args.config,
        source_override=args.source_override,
        storage_override=args.storage_override,
        working_dir_override=args.working_dir,
    )
    inv_df = _ensure_inventory_df(args, rel_cfg)
    adapter = create_source_adapter(rel_cfg.source)
    storage = create_storage_backend(rel_cfg.storage)
    manager = ReleaseManager(storage)
    plan = manager.plan(inv_df, rel_cfg, adapter=adapter, dry_run=args.dry_run)
    print(json.dumps(plan.summary, indent=2))
    return 0 if plan.feasible else 2


def _cmd_build(args: argparse.Namespace) -> int:
    rel_cfg = load_release_config(
        args.config,
        source_override=args.source_override,
        storage_override=args.storage_override,
        working_dir_override=args.working_dir,
    )
    inv_df = _ensure_inventory_df(args, rel_cfg)
    adapter = create_source_adapter(rel_cfg.source)
    storage = create_storage_backend(rel_cfg.storage)
    manager = ReleaseManager(storage)
    res = manager.build(inv_df, adapter=adapter, config=rel_cfg, resume=args.resume)
    out = {
        "release_id": res.release_id,
        "total_images": len(res.manifest_df),
        "shard_count": len(res.receipts),
        "reused_parent_shards": res.reused_parent_shards,
        "resumed_existing_shards": res.resumed_existing_shards,
        "newly_materialized_shards": res.newly_materialized_shards,
        "excluded_count": len(res.excluded_records),
        "metrics": res.metrics,
    }
    print(json.dumps(out, indent=2))
    return 0 if not res.excluded_records else 3


def _resolve_storage_for_release_cmd(args: argparse.Namespace) -> Any:
    if args.storage_config:
        return create_storage_backend(load_storage_config(args.storage_config))
    if args.config:
        rel_cfg = load_release_config(args.config)
        return create_storage_backend(rel_cfg.storage)
    raise ValueError("Provide either --storage-config or --config to locate storage backend.")


def _cmd_verify(args: argparse.Namespace) -> int:
    storage = _resolve_storage_for_release_cmd(args)
    release_id = args.release or load_release_config(args.config).release_id
    manager = ReleaseManager(storage)
    report = manager.verify(release_id, verify_tar_members=args.verify_tar_members)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def _cmd_publish(args: argparse.Namespace) -> int:
    storage = _resolve_storage_for_release_cmd(args)
    release_id = args.release or load_release_config(args.config).release_id
    manager = ReleaseManager(storage)
    meta = manager.publish(release_id, verify_tar_members=args.verify_tar_members)
    print(json.dumps(meta, indent=2))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    storage = _resolve_storage_for_release_cmd(args)
    release_id = args.release or load_release_config(args.config).release_id
    consumer = ReleaseConsumer(storage=storage, cache_dir=args.cache_dir)
    prepared = consumer.prepare_split(
        release_id=release_id,
        split=args.split,
        verify_checksums=not args.no_verify_checksums,
        max_cache_bytes=args.max_cache_bytes,
    )
    out = {
        "release_id": prepared.release_id,
        "split": prepared.split,
        "num_samples": len(prepared.manifest_df),
        "shard_count": len(prepared.shard_paths),
        "cache_hits": prepared.cache_hits,
        "cache_misses": prepared.cache_misses,
        "prepare_seconds": round(prepared.prepare_seconds, 4),
        "cache_dir": str(Path(args.cache_dir).resolve()),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    rel_cfg = load_release_config(
        args.config,
        source_override=args.source_override,
        storage_override=args.storage_override,
        working_dir_override=args.working_dir,
    )
    report = run_pipeline_benchmark(
        config=rel_cfg,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prooflens.data.cli",
        description="ProofLens Scalable GenImage Dataset Pipeline CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inventory
    p_inv = sub.add_parser(
        "inventory",
        help="Enumerate an upstream source into bounded-memory canonical Parquet inventory.",
    )
    inv_src = p_inv.add_mutually_exclusive_group(required=True)
    inv_src.add_argument("--config", type=Path, help="Path to release YAML config.")
    inv_src.add_argument("--source-config", type=Path, help="Path to standalone source YAML.")
    p_inv.add_argument("--output", type=Path, default=None, help="Output Parquet path.")
    p_inv.add_argument("--summary-out", type=Path, default=None, help="Output summary JSON path.")
    p_inv.add_argument("--batch-size", type=int, default=5000, help="Parquet writer batch size.")
    p_inv.add_argument(
        "--validate-bytes",
        action="store_true",
        help="Decode & SHA-256 hash image bytes during inventory (slower; full I/O).",
    )
    p_inv.add_argument("--source-override", type=Path, default=None)
    p_inv.add_argument("--storage-override", type=Path, default=None)
    p_inv.add_argument("--working-dir", type=str, default=None)
    p_inv.set_defaults(func=_cmd_inventory)

    # plan
    p_plan = sub.add_parser(
        "plan",
        help="Generate deterministic nested release plan and verify quota/leakage feasibility.",
    )
    p_plan.add_argument("--config", type=Path, required=True, help="Path to release YAML config.")
    p_plan.add_argument("--inventory", type=Path, default=None, help="Existing inventory Parquet.")
    p_plan.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate feasibility and print plan without uploading plan artifacts.",
    )
    p_plan.add_argument("--source-override", type=Path, default=None)
    p_plan.add_argument("--storage-override", type=Path, default=None)
    p_plan.add_argument("--working-dir", type=str, default=None)
    p_plan.set_defaults(func=_cmd_plan)

    # build
    p_build = sub.add_parser(
        "build",
        help="Materialize selected images into verified immutable TAR shards with resume support.",
    )
    p_build.add_argument("--config", type=Path, required=True, help="Path to release YAML config.")
    p_build.add_argument("--inventory", type=Path, default=None, help="Existing inventory Parquet.")
    p_build.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from verified committed shard receipts (default: enabled).",
    )
    p_build.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and require a fresh build.",
    )
    p_build.add_argument("--source-override", type=Path, default=None)
    p_build.add_argument("--storage-override", type=Path, default=None)
    p_build.add_argument("--working-dir", type=str, default=None)
    p_build.set_defaults(func=_cmd_build)

    # verify
    p_ver = sub.add_parser(
        "verify",
        help="Verify checksums, shard integrity, manifest schema, quotas, and leakage invariants.",
    )
    p_ver.add_argument("--release", type=str, default=None, help="Target release ID.")
    p_ver.add_argument("--config", type=Path, default=None, help="Release YAML config.")
    p_ver.add_argument("--storage-config", type=Path, default=None, help="Storage YAML config.")
    p_ver.add_argument(
        "--verify-tar-members",
        action="store_true",
        help="Download and inspect every TAR member header in addition to remote hash check.",
    )
    p_ver.set_defaults(func=_cmd_verify)

    # publish
    p_pub = sub.add_parser(
        "publish",
        help="Verify release invariants and publish immutable release.json last.",
    )
    p_pub.add_argument("--release", type=str, default=None, help="Target release ID.")
    p_pub.add_argument("--config", type=Path, default=None, help="Release YAML config.")
    p_pub.add_argument("--storage-config", type=Path, default=None, help="Storage YAML config.")
    p_pub.add_argument("--verify-tar-members", action="store_true")
    p_pub.set_defaults(func=_cmd_publish)

    # prepare
    p_prep = sub.add_parser(
        "prepare",
        help="Stage and verify shards of a published release into local runtime cache.",
    )
    p_prep.add_argument("--release", type=str, default=None, help="Published release ID.")
    p_prep.add_argument("--config", type=Path, default=None, help="Release YAML config.")
    p_prep.add_argument("--storage-config", type=Path, default=None, help="Storage YAML config.")
    p_prep.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split to stage (e.g. train, dev, seen_test, unseen_test; omit for all splits).",
    )
    p_prep.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("prooflens_cache"),
        help="Local cache directory (default: prooflens_cache).",
    )
    p_prep.add_argument(
        "--max-cache-bytes",
        type=int,
        default=None,
        help="Optional bounded LRU cache size in bytes.",
    )
    p_prep.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help="Skip SHA-256 re-verification of already cached shards (still checks size).",
    )
    p_prep.set_defaults(func=_cmd_prepare)

    # benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run end-to-end throughput and Python peak-memory profiling on a release config.",
    )
    p_bench.add_argument("--config", type=Path, required=True, help="Release YAML config.")
    p_bench.add_argument("--cache-dir", type=Path, default=Path("artifacts/bench_cache"))
    p_bench.add_argument("--split", type=str, default="train")
    p_bench.add_argument("--batch-size", type=int, default=8)
    p_bench.add_argument("--source-override", type=Path, default=None)
    p_bench.add_argument("--storage-override", type=Path, default=None)
    p_bench.add_argument("--working-dir", type=str, default=None)
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        err_payload = {
            "status": "ERROR",
            "command": args.command,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(err_payload, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
