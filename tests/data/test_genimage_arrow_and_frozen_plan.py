"""Tests for Phase 2: GenImageArrowAdapter, Frozen Plan Consumer, Preflight, Pilot, and E2E Materialization.

Covers:
- Arrow IPC File & IPC Stream reading across distinct train and test schemas
- Original binary image byte preservation (zero re-encoding) & exact SHA-256 verification
- Canonical locator validation (negative/out-of-range rows, traversal, revision/label/path/duplicate errors)
- Bounded LRU Arrow cache reuse, eviction, and proactive shard release
- Frozen snapshot validation (checksums, sizes, quotas, leakage, Plan ID preservation)
- Sampling vs materialization provenance separation & compatibility rules (never implicitly re-sampling)
- Storage capacity preflight (PASS, FAIL, UNKNOWN remote quota)
- Isolated pilot materialization & strict frozen failure classification
- Interrupted materialization, upload recovery, resume without duplicate commits
- End-to-end synthetic integration: Arrow -> Frozen Plan -> Materializer -> TAR -> LocalStorage -> Verify -> Publish -> Consumer (byte-for-byte match!)
- Explicitly opt-in real Hugging Face Arrow test (skipped offline by default)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest
import yaml
from PIL import Image

from prooflens.data.cli import main as cli_main
from prooflens.data.config import (
    ProtocolConfig,
    ReleaseConfig,
    SourceConfig,
    SplitRuleConfig,
    StorageConfig,
    load_release_config,
)
from prooflens.data.processing.materializer import (
    FrozenMaterializationError,
    MaterializationFailureCategory,
)
from prooflens.data.processing.shard_writer import compute_file_sha256
from prooflens.data.registry.schema import (
    LABEL_FAKE,
    LABEL_REAL,
    make_canonical_image_id,
    write_canonical_parquet,
)
from prooflens.data.releases.frozen_plan import (
    FrozenPlanCompatibilityError,
    FrozenSnapshotError,
    check_frozen_plan_compatibility,
    load_and_validate_frozen_snapshot,
)
from prooflens.data.releases.manager import ReleaseManager, load_release
from prooflens.data.releases.preflight import run_capacity_preflight
from prooflens.data.sampling.planner import plan_release
from prooflens.data.sources.base import SourceAccessError, SourceRecord
from prooflens.data.sources.genimage_arrow import (
    ArrowLocatorError,
    GenImageArrowAdapter,
    build_arrow_source_path,
    parse_arrow_shard_path,
    parse_arrow_source_path,
)
from prooflens.data.storage.google_drive import GoogleDriveStorage
from prooflens.data.storage.local import LocalFilesystemStorage


def _make_encoded_image_bytes(
    seed: int, fmt: str = "PNG", size: int = 24
) -> tuple[bytes, int, int]:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue(), size, size


def _write_arrow_shard(
    path: Path,
    rows: list[dict[str, Any]],
    include_train_optional_cols: bool,
    ipc_format: str = "stream",
) -> None:
    """Write a synthetic Arrow IPC stream or file matching verified GenImage-arrow schemas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        pa.field("image_path", pa.string()),
        pa.field("image", pa.binary()),
        pa.field("label", pa.int64()),
        pa.field("generator", pa.int64()),
    ]
    if include_train_optional_cols:
        fields = [
            pa.field("image_path", pa.string()),
            pa.field("md5", pa.string()),
            pa.field("width", pa.int64()),
            pa.field("height", pa.int64()),
            pa.field("image", pa.binary()),
            pa.field("label", pa.int64()),
            pa.field("generator", pa.int64()),
        ]
    schema = pa.schema(fields)

    # Split across 2 record batches when >= 2 rows to test multi-batch bounded streaming
    mid = max(1, len(rows) // 2)
    chunks = [rows[:mid], rows[mid:]] if len(rows) > 1 else [rows]

    def _to_batch(chunk: list[dict[str, Any]]) -> pa.RecordBatch:
        cols: dict[str, list[Any]] = {f.name: [r[f.name] for r in chunk] for f in fields}
        return pa.RecordBatch.from_pydict(cols, schema=schema)

    if ipc_format == "file":
        with pa.OSFile(str(path), "wb") as sink, pa_ipc.new_file(sink, schema) as writer:
            for ch in chunks:
                writer.write_batch(_to_batch(ch))
    else:
        with pa.OSFile(str(path), "wb") as sink, pa_ipc.new_stream(sink, schema) as writer:
            for ch in chunks:
                writer.write_batch(_to_batch(ch))


def _build_synthetic_arrow_fixture_and_snapshot(
    tmp_path: Path,
    source_revision: str = "3f4b9f921a673be09a93b335ed728cea0c6ecf33",
) -> dict[str, Any]:
    """Create a multi-shard synthetic Arrow source (both IPC file & stream, train & test schemas)
    plus an authentic frozen sampling snapshot (`freeze.json`, `plan.json`, `plan_selection.parquet`,
    `resolved_config.json`) preserving exact original image bytes for verification.
    """
    arrow_root = tmp_path / "arrow_repo"
    orig_bytes_by_id: dict[str, bytes] = {}
    inv_rows: list[dict[str, Any]] = []
    seed_val = 500

    # Shards covering seen (ADM, BigGAN) and unseen (Midjourney), train (IPC stream, full schema)
    # and test (IPC file, minimal schema without md5/width/height).
    shard_specs = [
        ("data/train/ADM/data-00000-of-00002.arrow", "train", "adm", 0, True, "stream", 10),
        ("data/train/BigGAN/data-00000-of-00002.arrow", "train", "biggan", 1, True, "file", 10),
        ("data/test/ADM/data-00000-of-00001.arrow", "test", "adm", 0, False, "file", 6),
        ("data/test/BigGAN/data-00000-of-00001.arrow", "test", "biggan", 1, False, "stream", 6),
        (
            "data/test/Midjourney/data-00000-of-00001.arrow",
            "test",
            "midjourney",
            2,
            False,
            "stream",
            6,
        ),
    ]

    for (
        shard_rel,
        split_name,
        gen_group,
        num_gen_code,
        has_opt,
        ipc_fmt,
        count_per_lbl,
    ) in shard_specs:
        shard_rows: list[dict[str, Any]] = []
        row_idx = 0
        for lbl in (LABEL_REAL, LABEL_FAKE):
            img_fmt = "JPEG" if lbl == LABEL_REAL else "PNG"
            ext = ".jpg" if lbl == LABEL_REAL else ".png"
            for i in range(count_per_lbl):
                raw_bytes, w, h = _make_encoded_image_bytes(seed_val, fmt=img_fmt, size=24)
                seed_val += 1
                orig_path = f"n01440764_{split_name}_{gen_group}_{lbl}_{i:03d}{ext}"
                md5_hex = hashlib.md5(raw_bytes).hexdigest()
                sha_hex = hashlib.sha256(raw_bytes).hexdigest()
                row_rec: dict[str, Any] = {
                    "image_path": orig_path,
                    "image": raw_bytes,
                    "label": lbl,
                    "generator": num_gen_code,
                }
                if has_opt:
                    row_rec["md5"] = md5_hex
                    row_rec["width"] = w
                    row_rec["height"] = h
                shard_rows.append(row_rec)

                canonical_source_path = build_arrow_source_path(shard_rel, row_idx, orig_path)
                img_id = make_canonical_image_id(
                    "genimage_arrow_sharded", source_revision, canonical_source_path
                )
                orig_bytes_by_id[img_id] = raw_bytes
                inv_rows.append(
                    {
                        "image_id": img_id,
                        "source_id": "genimage_arrow_sharded",
                        "source_revision": source_revision,
                        "source_path": canonical_source_path,
                        "source_split": split_name,
                        "label": lbl,
                        "generator": "none" if lbl == LABEL_REAL else gen_group,
                        "source_generator_group": gen_group,
                        "source_shard": shard_rel,
                        "imagenet_class": "n01440764",
                        "leakage_group_id": f"lg_sha_{sha_hex[:16]}",
                        "eligibility_status": "ELIGIBLE",
                        "width": w if has_opt else None,
                        "height": h if has_opt else None,
                        "image_format": img_fmt if has_opt else None,
                        "file_size_bytes": len(raw_bytes),
                        "sha256": sha_hex,
                        "source_row_index": row_idx,
                        "original_image_path": orig_path,
                    }
                )
                row_idx += 1

        _write_arrow_shard(
            arrow_root / shard_rel,
            shard_rows,
            include_train_optional_cols=has_opt,
            ipc_format=ipc_fmt,
        )

    inv_df = pd.DataFrame(inv_rows)

    # Create original sampling config (notice source_type="archive" to prove authorized adapter upgrade!)
    protocol = ProtocolConfig(
        protocol_version="cross_generator_v1",
        seen_generators=("adm", "biggan"),
        unseen_generators=("midjourney",),
        splits={
            "train": SplitRuleConfig(
                allowed_source_splits=("train",),
                generator_pool="seen",
                frozen_across_scaling=False,
                real_quota=8,
                fake_quota=8,
            ),
            "dev": SplitRuleConfig(
                allowed_source_splits=("train",),
                generator_pool="seen",
                frozen_across_scaling=True,
                real_quota=4,
                fake_quota=4,
            ),
            "seen_test": SplitRuleConfig(
                allowed_source_splits=("test",),
                generator_pool="seen",
                frozen_across_scaling=True,
                real_quota=4,
                fake_quota=4,
            ),
            "unseen_test": SplitRuleConfig(
                allowed_source_splits=("test",),
                generator_pool="unseen",
                frozen_across_scaling=True,
                real_quota=4,
                fake_quota=4,
            ),
        },
    )

    sampling_cfg = ReleaseConfig(
        release_id="genimage-synth-v1",
        sampling_seed=20260401,
        source=SourceConfig(
            source_id="genimage_arrow_sharded",
            source_type="archive",  # Original sampling config used a different adapter/runtime
            source_revision=source_revision,
            location="colab_notebook_acquisition",
        ),
        protocol=protocol,
        storage=StorageConfig(backend="local", root_path=str(tmp_path / "old_sampling_storage")),
        working_dir=str(tmp_path / "old_sampling_work"),
    )

    plan = plan_release(inv_df, sampling_cfg, dry_run=False)

    # Persist authentic frozen snapshot directory with artifact SHA-256 + sizes in freeze.json
    snap_dir = tmp_path / "frozen_snapshot"
    plan_json_p, sel_pq_p = plan.save(snap_dir)
    resolved_cfg_p = snap_dir / "resolved_config.json"
    resolved_cfg_p.write_text(json.dumps(sampling_cfg.to_dict(), indent=2), encoding="utf-8")

    artifacts_manifest = {}
    for p in (plan_json_p, sel_pq_p, resolved_cfg_p):
        artifacts_manifest[p.name] = {
            "sha256": compute_file_sha256(p),
            "size_bytes": p.stat().st_size,
        }

    freeze_meta = {
        "freeze_version": "1.0",
        "release_id": sampling_cfg.release_id,
        "plan_id": plan.plan_id,
        "sampling_config_hash": sampling_cfg.config_hash,
        "sampling_code_commit": "d76c8fd6105bfef029bcb2b8dbead255bf03ff5c",
        "source_id": "genimage_arrow_sharded",
        "source_revision": source_revision,
        "selected_total": len(plan.selection_df),
        "created_at_utc": "2026-11-24T13:15:04Z",
        "artifacts": artifacts_manifest,
    }
    (snap_dir / "freeze.json").write_text(json.dumps(freeze_meta, indent=2), encoding="utf-8")

    # Materialization config using new GenImageArrowAdapter & different runtime settings
    mat_storage_root = tmp_path / "prod_storage"
    mat_cfg = ReleaseConfig(
        release_id="genimage-synth-v1",
        sampling_seed=20260401,
        source=SourceConfig(
            source_id="genimage_arrow_sharded",
            source_type="huggingface_arrow",
            source_revision=source_revision,
            location="nebula/GenImage-arrow",
            arrow_local_root=str(arrow_root),
            arrow_cache_dir=str(tmp_path / "arrow_cache"),
            arrow_max_cached_shards=2,
            strict_generator_mapping=True,
        ),
        protocol=protocol,
        storage=StorageConfig(backend="local", root_path=str(mat_storage_root)),
        shard_target_bytes=512 * 1024,
        max_samples_per_shard=6,
        materialization_workers=2,
        working_dir=str(tmp_path / "mat_work"),
        strict_frozen_mode=True,
    )

    return {
        "arrow_root": arrow_root,
        "snap_dir": snap_dir,
        "plan": plan,
        "sampling_cfg": sampling_cfg,
        "mat_cfg": mat_cfg,
        "mat_storage_root": mat_storage_root,
        "orig_bytes_by_id": orig_bytes_by_id,
    }


def test_arrow_adapter_ipc_stream_file_validation_and_edge_cases(tmp_path: Path) -> None:
    fx = _build_synthetic_arrow_fixture_and_snapshot(tmp_path)
    arrow_root = fx["arrow_root"]
    rev = "3f4b9f921a673be09a93b335ed728cea0c6ecf33"

    # Locator & shard path parsing edge cases
    parsed_sh = parse_arrow_shard_path(
        "data/train/stable_diffusion_v_1_4/data-00001-of-00050.arrow"
    )
    assert parsed_sh.source_generator_group == "sdv4"
    assert parsed_sh.source_split == "train"

    for bad_shard in (
        "../escape/data-00000.arrow",
        "data/train/ADM/not_arrow.parquet",
        "data/unknown_split/ADM/data-00000.arrow",
        "data/train/UnknownGen/data-00000.arrow",
    ):
        with pytest.raises((ArrowLocatorError, ValueError)):
            parse_arrow_shard_path(bad_shard, strict_generator_mapping=True)

    # Support both full shard path and basename in source_path locator
    loc_full = parse_arrow_source_path(
        "data/train/ADM/data-00000-of-00002.arrow/rows/00000003/img.jpg",
        expected_source_shard="data/train/ADM/data-00000-of-00002.arrow",
        expected_row_index=3,
        expected_original_path="img.jpg",
    )
    assert loc_full.row_index == 3

    loc_base = parse_arrow_source_path(
        "data-00000-of-00002.arrow/rows/00000003/img.jpg",
        expected_source_shard="data/train/ADM/data-00000-of-00002.arrow",
        expected_row_index=3,
        expected_original_path="img.jpg",
    )
    assert loc_base.row_index == 3

    # Negative row index, traversal, original path mismatch rejected
    with pytest.raises(ArrowLocatorError):
        parse_arrow_source_path("data/train/ADM/data-00000.arrow/rows/-0000001/img.jpg")
    with pytest.raises(ValueError):
        parse_arrow_source_path("data/train/ADM/data-00000.arrow/rows/00000001/../secret.jpg")
    with pytest.raises(ArrowLocatorError):
        parse_arrow_source_path(
            "data/train/ADM/data-00000.arrow/rows/00000001/a.jpg",
            expected_original_path="b.jpg",
        )

    # Test adapter with injected shard_resolver to verify bounded LRU cache reuse & eviction
    download_calls: list[str] = []

    def _resolver(shard_rel: str) -> Path:
        download_calls.append(shard_rel)
        return arrow_root / shard_rel

    cfg = SourceConfig(
        source_id="genimage_arrow_sharded",
        source_type="huggingface_arrow",
        source_revision=rev,
        location="nebula/GenImage-arrow",
        arrow_cache_dir=str(tmp_path / "lru_cache"),
        arrow_max_cached_shards=2,
        strict_generator_mapping=True,
    )
    adapter = GenImageArrowAdapter(cfg, shard_resolver=_resolver)
    assert adapter.capabilities().random_access is False
    assert adapter.capabilities().shard_level_access is True

    # Enumerate via local_root adapter to test both IPC stream (ADM) and IPC file (BigGAN/test)
    local_adapter = GenImageArrowAdapter(
        SourceConfig(
            source_id="genimage_arrow_sharded",
            source_type="huggingface_arrow",
            source_revision=rev,
            location="nebula/GenImage-arrow",
            arrow_local_root=str(arrow_root),
            strict_generator_mapping=True,
        )
    )
    all_recs = list(local_adapter.iter_records())
    assert len(all_recs) == 76  # 20 + 20 + 12 + 12 + 12
    # Verify REAL provenance vs FAKE provenance (never numeric string "0"/"1")
    real_recs = [r for r in all_recs if r.label == LABEL_REAL]
    fake_recs = [r for r in all_recs if r.label == LABEL_FAKE]
    assert all(r.generator == "none" for r in real_recs)
    assert all(r.generator == r.source_generator_group for r in fake_recs)
    assert set(r.source_generator_group for r in all_recs) == {"adm", "biggan", "midjourney"}

    # Single-image fetch_image_bytes MUST be disabled
    with pytest.raises(SourceAccessError, match="disabled"):
        adapter.fetch_image_bytes(all_recs[0])

    # Retrieve multiple records from same shard -> only 1 download!
    adm_train_recs = [
        r for r in all_recs if r.source_shard == "data/train/ADM/data-00000-of-00002.arrow"
    ][:4]
    out1 = list(adapter.iter_selected_images(adm_train_recs))
    assert len(out1) == 4 and all(p is not None and e is None for _, p, e in out1)
    assert len(download_calls) == 1
    # Calling again reuses LRU cache (still 1 total download!)
    out2 = list(adapter.iter_selected_images(adm_train_recs[:2]))
    assert len(out2) == 2 and all(p is not None and e is None for _, p, e in out2)
    assert len(download_calls) == 1
    assert adapter.runtime_stats()["cache_hits"] >= 1

    # Negative / error cases: out-of-range row, revision mismatch, label mismatch, SHA mismatch,
    # missing shard, duplicate locator, invalid Arrow schema
    sample_rec = adm_train_recs[0]
    sample_rec_2 = adm_train_recs[1]
    bad_records = [
        SourceRecord(**{**sample_rec.__dict__, "source_revision": "wrong_rev"}),
        SourceRecord(
            **{
                **sample_rec.__dict__,
                "source_path": build_arrow_source_path(
                    sample_rec.source_shard or "", 9999, sample_rec.original_image_path or "x.jpg"
                ),
                "source_row_index": 9999,
            }
        ),
        SourceRecord(**{**sample_rec.__dict__, "label": 1 - sample_rec.label}),
        SourceRecord(**{**sample_rec_2.__dict__, "expected_sha256": "0" * 64}),
        SourceRecord(
            **{
                **sample_rec.__dict__,
                "source_shard": "data/train/ADM/data-99999-of-99999.arrow",
                "source_path": "data/train/ADM/data-99999-of-99999.arrow/rows/00000000/x.jpg",
                "source_row_index": 0,
                "original_image_path": "x.jpg",
            }
        ),
    ]
    bad_results = list(adapter.iter_selected_images(bad_records))
    assert len(bad_results) == len(bad_records)
    assert all(payload is None and err is not None for _, payload, err in bad_results)
    err_texts = [err or "" for _, _, err in bad_results]
    assert any("SOURCE_LOCATOR_MISMATCH" in t for t in err_texts)
    assert any("SHA256_MISMATCH" in t for t in err_texts)
    assert any("SOURCE_FETCH_FAILURE" in t for t in err_texts)

    # Duplicate locator in same batch rejected
    dup_res = list(adapter.iter_selected_images([sample_rec, sample_rec]))
    assert any(err and "Duplicate Arrow locator" in err for _, _, err in dup_res)

    # Invalid Arrow schema (missing mandatory `image` column) rejected cleanly
    bad_schema_shard = arrow_root / "data/train/ADM/data-00001-of-00002.arrow"
    bad_schema = pa.schema([pa.field("image_path", pa.string()), pa.field("label", pa.int64())])
    with pa.OSFile(str(bad_schema_shard), "wb") as sink, pa_ipc.new_stream(sink, bad_schema) as w:
        w.write_batch(
            pa.RecordBatch.from_pydict({"image_path": ["a.jpg"], "label": [0]}, bad_schema)
        )
    bad_schema_rec = SourceRecord(
        source_id="genimage_arrow_sharded",
        source_revision=rev,
        source_path="data/train/ADM/data-00001-of-00002.arrow/rows/00000000/a.jpg",
        source_split="train",
        label=0,
        generator="none",
        source_generator_group="adm",
        source_shard="data/train/ADM/data-00001-of-00002.arrow",
        source_row_index=0,
        original_image_path="a.jpg",
    )
    schema_res = list(local_adapter.iter_selected_images([bad_schema_rec]))
    assert schema_res[0][1] is None and "ArrowSchemaError" in (schema_res[0][2] or "")


def test_frozen_plan_validation_and_compatibility_rules(tmp_path: Path) -> None:
    fx = _build_synthetic_arrow_fixture_and_snapshot(tmp_path)
    snap_dir = fx["snap_dir"]
    mat_cfg: ReleaseConfig = fx["mat_cfg"]

    # 1. Valid frozen snapshot loads cleanly and preserves original Plan ID
    snapshot = load_and_validate_frozen_snapshot(snap_dir, expected_total_records=40)
    assert snapshot.sampling_provenance["sampling_plan_id"] == fx["plan"].plan_id
    assert snapshot.sampling_provenance["sampling_config_hash"] == fx["sampling_cfg"].config_hash
    assert snapshot.validation_report["plan_id_verified"] is True
    assert snapshot.validation_report["arrow_locators_verified"] is True

    # 2. Authorized runtime/adapter/storage changes pass compatibility check!
    compat = check_frozen_plan_compatibility(snapshot, mat_cfg)
    assert compat.compatible is True
    assert "source.source_type" in compat.authorized_runtime_differences
    assert (
        compat.sampling_provenance["sampling_config_hash"]
        != compat.materialization_provenance["materialization_config_hash"]
    )

    # 3. Unauthorized semantic changes MUST be rejected!
    from dataclasses import replace

    bad_rev_cfg = replace(mat_cfg, source=replace(mat_cfg.source, source_revision="deadbeef"))
    with pytest.raises(FrozenPlanCompatibilityError, match="source_revision"):
        check_frozen_plan_compatibility(snapshot, bad_rev_cfg)

    bad_seed_cfg = replace(mat_cfg, sampling_seed=9999)
    with pytest.raises(FrozenPlanCompatibilityError, match="sampling_seed"):
        check_frozen_plan_compatibility(snapshot, bad_seed_cfg)

    bad_proto_cfg = replace(
        mat_cfg, protocol=replace(mat_cfg.protocol, protocol_version="tampered_v2")
    )
    with pytest.raises(FrozenPlanCompatibilityError, match="protocol_version"):
        check_frozen_plan_compatibility(snapshot, bad_proto_cfg)

    # 4. Tampered artifact checksum MUST fail snapshot validation!
    tampered_dir = tmp_path / "tampered_snap"
    tampered_dir.mkdir()
    for p in snap_dir.iterdir():
        (tampered_dir / p.name).write_bytes(p.read_bytes())
    (tampered_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FrozenSnapshotError, match="SHA-256 mismatch|size mismatch"):
        load_and_validate_frozen_snapshot(tampered_dir)

    # 5. Duplicate SHA-256 or forged Plan ID MUST fail snapshot validation!
    bad_plan_dir = tmp_path / "bad_plan_id_snap"
    bad_plan_dir.mkdir()
    for p in snap_dir.iterdir():
        (bad_plan_dir / p.name).write_bytes(p.read_bytes())
    p_meta = json.loads((bad_plan_dir / "plan.json").read_text(encoding="utf-8"))
    p_meta["plan_id"] = "plan_0000000000000000"
    (bad_plan_dir / "plan.json").write_text(json.dumps(p_meta), encoding="utf-8")
    f_meta = json.loads((bad_plan_dir / "freeze.json").read_text(encoding="utf-8"))
    f_meta["artifacts"]["plan.json"] = {
        "sha256": compute_file_sha256(bad_plan_dir / "plan.json"),
        "size_bytes": (bad_plan_dir / "plan.json").stat().st_size,
    }
    (bad_plan_dir / "freeze.json").write_text(json.dumps(f_meta), encoding="utf-8")
    with pytest.raises(FrozenSnapshotError, match="Original Plan ID integrity check failed"):
        load_and_validate_frozen_snapshot(bad_plan_dir)

    # 6. Duplicate SHA-256 in selection rejected
    dup_sha_dir = tmp_path / "dup_sha_snap"
    dup_sha_dir.mkdir()
    for p in snap_dir.iterdir():
        (dup_sha_dir / p.name).write_bytes(p.read_bytes())
    sel_df = snapshot.selection_df.copy()
    sel_df.loc[1, "sha256"] = sel_df.loc[0, "sha256"]
    write_canonical_parquet(sel_df, dup_sha_dir / "plan_selection.parquet")
    f_meta2 = json.loads((dup_sha_dir / "freeze.json").read_text(encoding="utf-8"))
    f_meta2["artifacts"]["plan_selection.parquet"] = {
        "sha256": compute_file_sha256(dup_sha_dir / "plan_selection.parquet"),
        "size_bytes": (dup_sha_dir / "plan_selection.parquet").stat().st_size,
    }
    (dup_sha_dir / "freeze.json").write_text(json.dumps(f_meta2), encoding="utf-8")
    with pytest.raises(FrozenSnapshotError, match="Duplicate SHA-256"):
        load_and_validate_frozen_snapshot(dup_sha_dir)


def test_e2e_frozen_arrow_materialization_pilot_preflight_resume_and_byte_exactness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fx = _build_synthetic_arrow_fixture_and_snapshot(tmp_path)
    snap_dir: Path = fx["snap_dir"]
    mat_cfg: ReleaseConfig = fx["mat_cfg"]
    mat_storage_root: Path = fx["mat_storage_root"]
    orig_bytes_by_id: dict[str, bytes] = fx["orig_bytes_by_id"]

    # Prevent ANY implicit re-sampling during frozen build!
    def _forbid_resampling(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "plan_release() MUST NOT be called during frozen plan materialization!"
        )

    monkeypatch.setattr("prooflens.data.releases.manager.plan_release", _forbid_resampling)

    # Upload frozen snapshot into storage backend to exercise remote snapshot loading too
    storage = LocalFilesystemStorage(mat_storage_root)
    remote_prefix = "pipeline_metadata/sampling_freeze/rev1/20261124_131504"
    for p in snap_dir.iterdir():
        storage.upload(p, f"{remote_prefix}/{p.name}", overwrite=True)

    # Write YAML config for CLI end-to-end tests
    cfg_yaml = tmp_path / "mat_release.yaml"
    cfg_dict = mat_cfg.to_dict()
    cfg_dict["frozen_snapshot_prefix"] = remote_prefix
    cfg_yaml.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    # 1. CLI validate-frozen
    assert cli_main(["validate-frozen", "--config", str(cfg_yaml), "--expected-records", "40"]) == 0
    vf_out = json.loads(capsys.readouterr().out)
    assert vf_out["validation_report"]["valid"] is True
    assert vf_out["compatibility"]["compatible"] is True

    # 2. CLI preflight (local storage -> KNOWN & PASS)
    assert cli_main(["preflight", "--config", str(cfg_yaml)]) == 0
    pre_out = json.loads(capsys.readouterr().out)
    assert pre_out["overall_status"] == "PASS"
    assert pre_out["remote_status"] == "PASS"
    assert pre_out["local_status"] == "PASS"

    # Also verify preflight reports UNKNOWN + REQUIRES_EXPLICIT_APPROVAL when Drive quota unavailable
    class _UnknownQuotaService:
        def about(self) -> Any:
            class _Get:
                def get(self, **kwargs: Any) -> Any:
                    class _Exec:
                        def execute(self) -> dict[str, Any]:
                            return {"storageQuota": {"usage": "12345"}}  # no finite limit

                    return _Exec()

            return _Get()

    gdrive_unknown = GoogleDriveStorage(
        mode="api",
        folder_id="1IipfeWX_OVJS2KwaFL5wFlO-edxLoEGD",
        drive_service=_UnknownQuotaService(),
    )
    snap = load_and_validate_frozen_snapshot(snap_dir)
    pre_unknown = run_capacity_preflight(snap.selection_df, mat_cfg, gdrive_unknown)
    assert pre_unknown.remote_status == "UNKNOWN"
    assert pre_unknown.overall_status == "REQUIRES_EXPLICIT_APPROVAL"
    assert pre_unknown.requires_explicit_approval is True

    # 3. CLI pilot -> isolated scratch directory, official releases/ remains untouched!
    pilot_scratch = tmp_path / "pilot_scratch"
    assert (
        cli_main(
            [
                "pilot",
                "--config",
                str(cfg_yaml),
                "--sample-count",
                "12",
                "--pilot-storage-dir",
                str(pilot_scratch),
            ]
        )
        == 0
    )
    pilot_out = json.loads(capsys.readouterr().out)
    assert pilot_out["is_pilot"] is True
    assert pilot_out["official_release"] is False
    assert pilot_out["passed"] is True
    assert pilot_out["pilot_samples_materialized"] == 12
    assert pilot_out["pilot_readback_verified"] == 12
    assert pilot_out["pilot_real_count"] > 0 and pilot_out["pilot_fake_count"] > 0
    assert len(pilot_out["pilot_generator_groups"]) >= 2
    assert len(pilot_out["pilot_release_splits"]) >= 2
    assert pilot_out["pilot_unique_source_shards"] >= 2
    assert not storage.exists("releases/genimage-synth-v1/release.json")

    # 4. CLI build --frozen-snapshot --strict-frozen --resume (full synthetic 40-image release)
    assert cli_main(["build", "--config", str(cfg_yaml), "--strict-frozen", "--resume"]) == 0
    b_out = json.loads(capsys.readouterr().out)
    assert b_out["mode"] == "frozen_snapshot"
    assert b_out["sampling_plan_id"] == fx["plan"].plan_id
    assert b_out["total_images"] == 40
    assert b_out["excluded_count"] == 0

    # 5. Simulate interrupted materialization (corrupt 1 uncommitted remote shard) + resume without duplicates
    train_objs = sorted(storage.list_objects("releases/genimage-synth-v1/shards/train"))
    tar_key = next(k for k in train_objs if k.endswith(".tar"))
    (storage.root_dir / tar_key).write_bytes(b"corrupted_incomplete_shard")

    assert cli_main(["build", "--config", str(cfg_yaml), "--strict-frozen", "--resume"]) == 0
    r_out = json.loads(capsys.readouterr().out)
    assert r_out["newly_materialized_shards"] == 1
    assert r_out["resumed_existing_shards"] == b_out["shard_count"] - 1
    assert r_out["total_images"] == 40

    # 6. Verify + Publish
    assert cli_main(["verify", "--config", str(cfg_yaml), "--verify-tar-members"]) == 0
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["passed"] is True

    assert cli_main(["publish", "--config", str(cfg_yaml)]) == 0
    pub_out = json.loads(capsys.readouterr().out)
    loaded_yaml_cfg = load_release_config(cfg_yaml)
    assert pub_out["status"] == "PUBLISHED"
    assert pub_out["sampling_plan_id"] == fx["plan"].plan_id
    assert pub_out["sampling_config_hash"] == fx["sampling_cfg"].config_hash
    assert pub_out["materialization_config_hash"] == loaded_yaml_cfg.config_hash
    assert (
        pub_out["sampling_provenance"]["sampling_code_commit"]
        == "d76c8fd6105bfef029bcb2b8dbead255bf03ff5c"
    )

    # 7. Consumer read-back: verify EVERY recovered raw byte matches original Arrow fixture byte-for-byte!
    consumer_cache = tmp_path / "consumer_cache"
    recovered_count = 0
    for split_name in ("train", "dev", "seen_test", "unseen_test"):
        ds = load_release(
            release="genimage-synth-v1",
            split=split_name,
            cache_dir=consumer_cache,
            storage=storage,
            transform=None,
        )
        try:
            for idx in range(len(ds)):
                row = ds.manifest.iloc[idx]
                img_id = str(row["image_id"])
                shard_path = ds.shard_paths[str(row["shard_id"])]
                with shard_path.open("rb") as fh:
                    fh.seek(int(row["shard_offset_data"]))
                    recovered_bytes = fh.read(int(row["file_size_bytes"]))
                assert recovered_bytes == orig_bytes_by_id[img_id], (
                    f"Stored TAR bytes diverged from original Arrow bytes for {img_id}!"
                )
                sample = ds[idx]
                assert sample["image_id"] == img_id
                assert int(sample["label"].item()) == int(row["label"])
                recovered_count += 1
        finally:
            ds.close()

    assert recovered_count == 40


def test_strict_frozen_mode_fails_loudly_on_hash_mismatch_and_tracks_download_reuse(
    tmp_path: Path,
) -> None:
    fx = _build_synthetic_arrow_fixture_and_snapshot(tmp_path)
    snap = load_and_validate_frozen_snapshot(fx["snap_dir"])
    mat_cfg: ReleaseConfig = fx["mat_cfg"]
    arrow_root: Path = fx["arrow_root"]

    # Inject a shard_resolver counting downloads across all splits (dev + train share ADM/BigGAN Arrow shards!)
    download_log: list[str] = []

    def _counting_resolver(shard_rel: str) -> Path:
        download_log.append(shard_rel)
        return arrow_root / shard_rel

    from dataclasses import replace

    lru_cfg = replace(
        mat_cfg,
        source=replace(
            mat_cfg.source,
            arrow_local_root=None,
            arrow_cache_dir=str(tmp_path / "dl_reuse_cache"),
            arrow_max_cached_shards=5,  # large enough to keep shared train shards across dev -> train
        ),
    )
    adapter = GenImageArrowAdapter(lru_cfg.source, shard_resolver=_counting_resolver)
    storage = LocalFilesystemStorage(tmp_path / "reuse_storage")
    manager = ReleaseManager(storage)
    mat_res, _ = manager.build_from_frozen_snapshot(snap, adapter, lru_cfg, strict_frozen_mode=True)
    assert len(mat_res.manifest_df) == 40
    # Exactly 5 distinct Arrow source shards -> each downloaded ONCE despite dev + train sharing ADM & BigGAN!
    assert len(download_log) == 5
    assert adapter.runtime_stats()["downloads"] == 5
    # All managed shards proactively released after final reference completes
    assert adapter.runtime_stats()["cached_shards_current"] == 0

    # Now tamper one frozen row's expected sha256 -> strict_frozen_mode MUST raise FrozenMaterializationError
    bad_df = snap.selection_df.copy()
    tampered_id = str(bad_df.iloc[0]["image_id"])
    bad_df.loc[0, "sha256"] = "f" * 64
    from prooflens.data.processing.materializer import materialize_release

    storage_bad = LocalFilesystemStorage(tmp_path / "bad_hash_storage")
    adapter_bad = GenImageArrowAdapter(mat_cfg.source)
    with pytest.raises(FrozenMaterializationError) as exc_info:
        materialize_release(bad_df, adapter_bad, storage_bad, mat_cfg, strict_frozen_mode=True)
    assert exc_info.value.image_id == tampered_id
    assert exc_info.value.category == MaterializationFailureCategory.SHA256_MISMATCH


def test_near_duplicate_quality_gate_blocks_or_warns_without_dropping(tmp_path: Path) -> None:
    fx = _build_synthetic_arrow_fixture_and_snapshot(tmp_path)
    snap = load_and_validate_frozen_snapshot(fx["snap_dir"])
    mat_cfg: ReleaseConfig = fx["mat_cfg"]

    from dataclasses import replace

    gate_cfg = replace(mat_cfg, near_duplicate_quality_gate="block_publish")
    storage = LocalFilesystemStorage(tmp_path / "gate_storage")
    adapter = GenImageArrowAdapter(gate_cfg.source)
    manager = ReleaseManager(storage)
    manager.build_from_frozen_snapshot(snap, adapter, gate_cfg, strict_frozen_mode=True)

    # Inject a cross-split perceptual dHash candidate into audit_summary.json to test gate enforcement
    audit_key = f"releases/{gate_cfg.release_id}/audit_summary.json"
    audit_meta = json.loads(storage.read_bytes(audit_key).decode("utf-8"))
    audit_meta["materialization_metrics"]["cross_split_perceptual_candidates"] = 2
    storage.write_bytes(json.dumps(audit_meta, indent=2).encode("utf-8"), audit_key, overwrite=True)

    ver_block = manager.verify(gate_cfg.release_id)
    assert ver_block.passed is False
    assert ver_block.checks["near_duplicate_quality_gate"] is False

    # Under "warn" gate, verification passes with an explicit warning (never silently drops images)
    audit_meta["resolved_config"]["near_duplicate_quality_gate"] = "warn"
    storage.write_bytes(json.dumps(audit_meta, indent=2).encode("utf-8"), audit_key, overwrite=True)
    ver_warn = manager.verify(gate_cfg.release_id)
    assert ver_warn.passed is True
    assert ver_warn.checks["near_duplicate_quality_gate"] is True
    assert any("perceptual dHash" in w for w in ver_warn.warnings)


def test_production_arrow_yaml_configs_load_cleanly() -> None:
    cfg = load_release_config("configs/releases/genimage_120k_v1_arrow.yaml")
    assert cfg.release_id == "genimage-120k-v1"
    assert cfg.sampling_seed == 20260401
    assert cfg.source.source_id == "genimage_arrow_sharded"
    assert cfg.source.source_type == "huggingface_arrow"
    assert cfg.source.location == "nebula/GenImage-arrow"
    assert cfg.source.source_revision == "3f4b9f921a673be09a93b335ed728cea0c6ecf33"
    assert cfg.strict_frozen_mode is True
    assert cfg.frozen_snapshot_prefix is not None
    assert cfg.storage.gdrive_folder_id == "1IipfeWX_OVJS2KwaFL5wFlO-edxLoEGD"


@pytest.mark.skipif(
    os.environ.get("PROOFLENS_REAL_HF_ARROW_TEST") != "1",
    reason="Opt-in bounded real Hugging Face Arrow test; set PROOFLENS_REAL_HF_ARROW_TEST=1 on Colab.",
)
def test_real_hf_arrow_opt_in(tmp_path: Path) -> None:
    """Bounded opt-in verification against pinned `nebula/GenImage-arrow` revision (lists shards only)."""
    cfg = load_release_config("configs/releases/genimage_120k_v1_arrow.yaml")
    adapter = GenImageArrowAdapter(cfg.source)
    shards = adapter.list_available_shards()
    assert len(shards) > 0
    assert any(s.startswith("data/train/ADM/") for s in shards)
    assert any(s.startswith("data/test/Midjourney/") for s in shards)
