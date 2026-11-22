"""End-to-end integration tests for shards, storage backends, resume, nested releases, and PyTorch reading.

Also contains an opt-in real GenImage integration test (`test_real_genimage_opt_in`)
that runs ONLY when `PROOFLENS_REAL_GENIMAGE_ROOT` is explicitly set (e.g., on Colab).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image
from torchvision import transforms

from prooflens.data.benchmark import run_pipeline_benchmark
from prooflens.data.cli import main as cli_main
from prooflens.data.config import load_release_config
from prooflens.data.processing.shard_writer import (
    ImmutableTarShardWriter,
    ShardImmutabilityError,
    verify_tar_shard_file,
)
from prooflens.data.releases.manager import (
    BoundedShardCache,
    ReleaseConsumer,
    ReleaseManager,
    UnpublishedReleaseError,
    load_release,
)
from prooflens.data.shard_dataset import (
    CorruptedShardMemberError,
    ShardIterableDataset,
    ShardMapDataset,
    create_shard_dataloader,
)
from prooflens.data.sources.genimage import create_source_adapter
from prooflens.data.storage.base import ImmutabilityError
from prooflens.data.storage.google_drive import GoogleDriveStorage
from prooflens.data.storage.local import LocalFilesystemStorage


def _build_tiny_genimage_fixture(
    root: Path,
    train_per_gen_per_label: int = 8,
    val_per_gen_per_label: int = 4,
) -> None:
    """Create a tiny multi-generator directory fixture following GenImage hierarchy."""
    seed_counter = 100
    for gen_folder in ("imagenet_ai_0419_biggan", "imagenet_ai_0424_sdv5", "imagenet_midjourney"):
        for split, n_imgs in (("train", train_per_gen_per_label), ("val", val_per_gen_per_label)):
            for cls_folder, fmt, ext in (("nature", "JPEG", ".jpg"), ("ai", "PNG", ".png")):
                target = root / gen_folder / split / cls_folder
                target.mkdir(parents=True, exist_ok=True)
                for idx in range(n_imgs):
                    rng = np.random.default_rng(seed_counter)
                    seed_counter += 1
                    arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(
                        target / f"n01440764_{split}_{cls_folder}_{idx:03d}{ext}", format=fmt
                    )


def _write_test_yaml_suite(
    base_dir: Path,
    source_root: Path,
    storage_root: Path,
    work_root: Path,
) -> tuple[Path, Path, Path]:
    src_yaml = base_dir / "source.yaml"
    proto_yaml = base_dir / "protocol.yaml"
    store_yaml = base_dir / "storage.yaml"
    rel_small_yaml = base_dir / "release_small.yaml"
    rel_expanded_yaml = base_dir / "release_expanded.yaml"

    src_yaml.write_text(
        yaml.safe_dump(
            {
                "source": {
                    "source_id": "tiny_genimage_fixture",
                    "source_type": "directory",
                    "source_revision": "fixture-rev-1",
                    "location": str(source_root),
                    "generator_mapping": {
                        "imagenet_ai_0419_biggan": "biggan",
                        "imagenet_ai_0424_sdv5": "sdv5",
                        "imagenet_midjourney": "midjourney",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    proto_yaml.write_text(
        yaml.safe_dump(
            {
                "protocol": {
                    "protocol_version": "test_cross_gen_v1",
                    "seen_generators": ["biggan", "sdv5"],
                    "unseen_generators": ["midjourney"],
                    "splits": {
                        "train": {
                            "allowed_source_splits": ["train"],
                            "generator_pool": "seen",
                            "frozen_across_scaling": False,
                            "real_quota": 4,
                            "fake_quota": 4,
                        },
                        "dev": {
                            "allowed_source_splits": ["train"],
                            "generator_pool": "seen",
                            "frozen_across_scaling": True,
                            "real_quota": 2,
                            "fake_quota": 2,
                        },
                        "seen_test": {
                            "allowed_source_splits": ["val"],
                            "generator_pool": "seen",
                            "frozen_across_scaling": True,
                            "real_quota": 2,
                            "fake_quota": 2,
                        },
                        "unseen_test": {
                            "allowed_source_splits": ["val"],
                            "generator_pool": "unseen",
                            "frozen_across_scaling": True,
                            "real_quota": 2,
                            "fake_quota": 2,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    store_yaml.write_text(
        yaml.safe_dump({"storage": {"backend": "local", "root_path": str(storage_root)}}),
        encoding="utf-8",
    )
    rel_small_yaml.write_text(
        yaml.safe_dump(
            {
                "release_id": "small-release-v1",
                "sampling_seed": 42,
                "parent_release_id": None,
                "source_config": str(src_yaml),
                "protocol_config": str(proto_yaml),
                "storage_config": str(store_yaml),
                "shard_target_bytes": 1024 * 1024,
                "max_samples_per_shard": 4,
                "materialization_workers": 2,
                "working_dir": str(work_root),
            }
        ),
        encoding="utf-8",
    )
    rel_expanded_yaml.write_text(
        yaml.safe_dump(
            {
                "release_id": "expanded-release-v1",
                "sampling_seed": 42,
                "parent_release_id": "small-release-v1",
                "source_config": str(src_yaml),
                "protocol_config": str(proto_yaml),
                "storage_config": str(store_yaml),
                "split_quotas": {"train": {"real_quota": 8, "fake_quota": 8}},
                "shard_target_bytes": 1024 * 1024,
                "max_samples_per_shard": 4,
                "materialization_workers": 2,
                "working_dir": str(work_root),
            }
        ),
        encoding="utf-8",
    )
    return rel_small_yaml, rel_expanded_yaml, store_yaml


def test_immutable_tar_shard_writer_and_corruption_detection(tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(buf, format="PNG")
    raw = buf.getvalue()
    sha = hashlib.sha256(raw).hexdigest()

    writer = ImmutableTarShardWriter(
        shard_dir=tmp_path,
        shard_id="train-00000-test",
        release_id="rel-1",
        release_split="train",
        storage_key="releases/rel-1/shards/train/train-00000-test.tar",
    )
    writer.add_image(
        image_id="pl_abc",
        raw_bytes=raw,
        file_extension=".png",
        sha256=sha,
        label=1,
        generator="biggan",
        source_generator_group="biggan",
        source_split="train",
        width=16,
        height=16,
        image_format="PNG",
    )
    tar_path, receipt_path, receipt = writer.finalize()
    assert tar_path.is_file() and receipt_path.is_file()
    ok, msg = verify_tar_shard_file(
        tar_path,
        expected_sha256=receipt.sha256,
        expected_byte_size=receipt.byte_size,
        expected_image_count=1,
    )
    assert ok, msg

    # Attempting to overwrite a committed shard must raise ShardImmutabilityError
    with pytest.raises(ShardImmutabilityError):
        ImmutableTarShardWriter(
            shard_dir=tmp_path,
            shard_id="train-00000-test",
            release_id="rel-1",
            release_split="train",
            storage_key="releases/rel-1/shards/train/train-00000-test.tar",
        )


def test_e2e_cli_parent_child_resume_cache_and_pytorch_loaders(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "source_data"
    storage_root = tmp_path / "storage"
    work_root = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    _build_tiny_genimage_fixture(source_root)
    rel_small_yaml, rel_expanded_yaml, store_yaml = _write_test_yaml_suite(
        tmp_path, source_root, storage_root, work_root
    )

    # 1. CLI inventory
    assert cli_main(["inventory", "--config", str(rel_small_yaml)]) == 0
    inv_out = json.loads(capsys.readouterr().out)
    assert inv_out["total_records"] == 72

    # 2. CLI plan --dry-run (must not publish or upload plan artifacts)
    assert cli_main(["plan", "--config", str(rel_small_yaml), "--dry-run"]) == 0
    dry_plan = json.loads(capsys.readouterr().out)
    assert dry_plan["feasible"] is True
    assert dry_plan["selected_total"] == 20
    storage = LocalFilesystemStorage(storage_root)
    assert not storage.exists("releases/small-release-v1/plan.json")

    # 3. CLI build (materializes shards, does NOT publish yet)
    assert cli_main(["build", "--config", str(rel_small_yaml), "--resume"]) == 0
    build_out = json.loads(capsys.readouterr().out)
    assert build_out["total_images"] == 20
    assert (
        build_out["newly_materialized_shards"] == 5
    )  # train(8)=2 shards, dev(4)=1, seen(4)=1, unseen(4)=1

    # Consumer MUST reject unpublished release before `publish`
    consumer = ReleaseConsumer(storage=storage, cache_dir=cache_dir)
    with pytest.raises(UnpublishedReleaseError):
        consumer.get_published_release("small-release-v1")

    # 4. Simulate interrupted shard recovery on an unpublished release: corrupt 1 shard and re-run build --resume
    train_shards = sorted(storage.list_objects("releases/small-release-v1/shards/train"))
    tar_key = next(k for k in train_shards if k.endswith(".tar"))
    # Corrupt the uncommitted remote tar bytes directly on disk
    (storage.root_dir / tar_key).write_bytes(b"corrupted_partial_upload")
    assert cli_main(["build", "--config", str(rel_small_yaml), "--resume"]) == 0
    recover_out = json.loads(capsys.readouterr().out)
    assert recover_out["resumed_existing_shards"] == 4
    assert recover_out["newly_materialized_shards"] == 1

    # 5. CLI verify + publish
    assert (
        cli_main(
            [
                "verify",
                "--release",
                "small-release-v1",
                "--storage-config",
                str(store_yaml),
                "--verify-tar-members",
            ]
        )
        == 0
    )
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["passed"] is True

    assert (
        cli_main(["publish", "--release", "small-release-v1", "--storage-config", str(store_yaml)])
        == 0
    )
    pub_out = json.loads(capsys.readouterr().out)
    assert pub_out["status"] == "PUBLISHED"

    # Re-running build --resume on already published release must be idempotent no-op
    assert cli_main(["build", "--config", str(rel_small_yaml), "--resume"]) == 0
    idem_out = json.loads(capsys.readouterr().out)
    assert idem_out["newly_materialized_shards"] == 0
    assert idem_out["metrics"].get("idempotent_published_skip") is True

    # Attempting to overwrite published release artifact must raise ImmutabilityError
    with pytest.raises(ImmutabilityError):
        storage.write_bytes(b"hack", "releases/small-release-v1/release.json", overwrite=True)

    # 6. CLI prepare (1st call = cache misses, 2nd call = 100% cache hits)
    assert (
        cli_main(
            [
                "prepare",
                "--release",
                "small-release-v1",
                "--storage-config",
                str(store_yaml),
                "--split",
                "train",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    prep1 = json.loads(capsys.readouterr().out)
    assert prep1["cache_misses"] == 2 and prep1["cache_hits"] == 0

    assert (
        cli_main(
            [
                "prepare",
                "--release",
                "small-release-v1",
                "--storage-config",
                str(store_yaml),
                "--split",
                "train",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    prep2 = json.loads(capsys.readouterr().out)
    assert prep2["cache_misses"] == 0 and prep2["cache_hits"] == 2

    # 7. Nested Child Release Expansion (small-release-v1 -> expanded-release-v1)
    assert cli_main(["build", "--config", str(rel_expanded_yaml), "--resume"]) == 0
    child_build = json.loads(capsys.readouterr().out)
    assert child_build["total_images"] == 28  # 16 train + 4 dev + 4 seen + 4 unseen
    assert child_build["reused_parent_shards"] == 5  # ALL 5 parent shards reused!
    assert (
        child_build["newly_materialized_shards"] == 2
    )  # ONLY +8 delta train images -> 2 new shards!

    assert (
        cli_main(
            [
                "publish",
                "--release",
                "expanded-release-v1",
                "--storage-config",
                str(store_yaml),
                "--verify-tar-members",
            ]
        )
        == 0
    )
    child_pub = json.loads(capsys.readouterr().out)
    assert child_pub["parent_release_id"] == "small-release-v1"
    assert child_pub["verification_checks"]["parent_nested_inheritance"] is True

    # Verify parent/child ID invariants directly via consumer
    parent_manifest = consumer.load_manifest("small-release-v1")
    child_manifest = consumer.load_manifest("expanded-release-v1")
    p_train = set(parent_manifest.loc[parent_manifest["release_split"] == "train", "image_id"])
    c_train = set(child_manifest.loc[child_manifest["release_split"] == "train", "image_id"])
    assert p_train < c_train and len(p_train) == 8 and len(c_train) == 16
    for sp in ("dev", "seen_test", "unseen_test"):
        assert set(parent_manifest.loc[parent_manifest["release_split"] == sp, "image_id"]) == set(
            child_manifest.loc[child_manifest["release_split"] == sp, "image_id"]
        )

    # 8. Consumer PyTorch reading on child release (reads BOTH parent shards + new child shards!)
    # Note: source_root is NOT passed to load_release — consumer never touches original source!
    tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    map_ds = load_release(
        release="expanded-release-v1",
        split="train",
        cache_dir=cache_dir,
        storage=storage,
        transform=tfm,
    )
    assert isinstance(map_ds, ShardMapDataset)
    assert len(map_ds) == 16

    loader = create_shard_dataloader(map_ds, batch_size=4, shuffle=False, num_workers=0)
    seen_ids = []
    labels = []
    for batch in loader:
        assert batch["image"].shape == (4, 3, 32, 32)
        seen_ids.extend(batch["image_id"])
        labels.extend(batch["label"].int().tolist())
    assert len(seen_ids) == 16 and len(set(seen_ids)) == 16
    assert labels.count(0) == 8 and labels.count(1) == 8
    map_ds.close()

    # 9. ShardIterableDataset + BoundedShardCache LRU eviction + deterministic epoch count
    bounded_cache_dir = tmp_path / "bounded_cache"
    iter_ds = load_release(
        release="expanded-release-v1",
        split="train",
        cache_dir=bounded_cache_dir,
        storage=storage,
        transform=tfm,
        streaming=True,
        max_cache_bytes=12_000,  # forces LRU eviction across the 4 train shards
        shuffle_shards=True,
        seed=99,
    )
    assert isinstance(iter_ds, ShardIterableDataset)
    assert len(iter_ds) == 16
    assert isinstance(iter_ds.shard_cache, BoundedShardCache)
    iter_loader = create_shard_dataloader(iter_ds, batch_size=4, num_workers=0)
    iter_ids = []
    for batch in iter_loader:
        iter_ids.extend(batch["image_id"])
    assert set(iter_ids) == c_train
    assert iter_ds.shard_cache.evictions >= 1

    # 10. Multi-worker DataLoader correctness (num_workers=2 -> exact sample count, zero duplicates)
    mw_ds = load_release(
        release="expanded-release-v1",
        split="train",
        cache_dir=cache_dir,
        storage=storage,
        transform=tfm,
        streaming=True,
    )
    mw_loader = create_shard_dataloader(mw_ds, batch_size=4, num_workers=2)
    mw_ids = []
    for batch in mw_loader:
        mw_ids.extend(batch["image_id"])
    assert len(mw_ids) == 16 and set(mw_ids) == c_train

    # 11. Corrupted shard member must fail visibly (never silently skip)
    bad_cache = tmp_path / "corrupt_cache"
    bad_ds = load_release(
        release="small-release-v1",
        split="dev",
        cache_dir=bad_cache,
        storage=storage,
        transform=tfm,
    )
    # Tamper image bytes inside staged dev TAR
    dev_tar = next(iter(bad_ds.shard_paths.values()))
    data = bytearray(dev_tar.read_bytes())
    offset = int(bad_ds.manifest.iloc[0]["shard_offset_data"])
    data[offset : offset + 16] = b"\x00" * 16
    dev_tar.write_bytes(bytes(data))
    with pytest.raises(CorruptedShardMemberError):
        _ = bad_ds[0]
    bad_ds.close()


def test_corrupted_source_image_recorded_and_blocks_publish(tmp_path: Path) -> None:
    source_root = tmp_path / "corrupt_src"
    storage_root = tmp_path / "storage"
    work_root = tmp_path / "work"
    _build_tiny_genimage_fixture(source_root, train_per_gen_per_label=4, val_per_gen_per_label=2)
    rel_small_yaml, _, _ = _write_test_yaml_suite(tmp_path, source_root, storage_root, work_root)

    cfg = load_release_config(rel_small_yaml)
    adapter = create_source_adapter(cfg.source)
    storage = LocalFilesystemStorage(storage_root)
    manager = ReleaseManager(storage)

    # Corrupt ALL train/ai files in biggan so selection hits a corrupted file
    for bad_f in (source_root / "imagenet_ai_0419_biggan/train/ai").glob("*.png"):
        bad_f.write_bytes(b"not_a_real_png")

    from prooflens.data.registry.inventory import build_source_inventory
    from prooflens.data.registry.schema import read_canonical_parquet

    inv_res = build_source_inventory(adapter, tmp_path / "inv.parquet")
    inv_df = read_canonical_parquet(inv_res.parquet_path)

    mat_res = manager.build(inv_df, adapter, cfg, resume=True)
    assert len(mat_res.excluded_records) >= 1
    assert (work_root / "small-release-v1" / "exclusions.jsonl").is_file()

    # Verification must fail and block publication until explicit replanning!
    ver = manager.verify("small-release-v1")
    assert ver.passed is False
    assert ver.checks["zero_unresolved_exclusions"] is False


def test_google_drive_mounted_and_fake_api_modes(tmp_path: Path) -> None:
    # 1. Mounted mode with staged copy + fsync + read-back SHA-256 verification
    mount_root = tmp_path / "drive_mount"
    gdrive_mounted = GoogleDriveStorage(
        mode="mounted",
        folder_id="1IipfeWX_OVJS2KwaFL5wFlO-edxLoEGD",
        mount_root=mount_root,
    )
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"prooflens-gdrive-mounted-payload")
    meta = gdrive_mounted.upload(sample, "releases/r1/shards/train/s0.tar")
    assert gdrive_mounted.verify_integrity(
        "releases/r1/shards/train/s0.tar", expected_sha256=meta.sha256
    )
    with pytest.raises(ImmutabilityError):
        gdrive_mounted.upload(sample, "releases/r1/shards/train/s0.tar", overwrite=False)

    # 2. API v3 mode tested with lightweight in-memory fake Drive v3 service (no cloud creds needed)
    class _FakeExec:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def execute(self) -> Any:
            return self._payload

    class _FakeFilesResource:
        def __init__(self) -> None:
            self.items: dict[str, dict[str, Any]] = {}
            self.blobs: dict[str, bytes] = {}
            self.seq = 0

        def list(self, q: str = "", **kwargs: Any) -> _FakeExec:
            matches = []
            for item in self.items.values():
                if f"'{item['parent']}' in parents" not in q:
                    continue
                if "name = '" in q:
                    want_name = q.split("name = '", 1)[1].split("'", 1)[0]
                    if item["name"] != want_name:
                        continue
                if "mimeType = '" in q:
                    want_mime = q.split("mimeType = '", 1)[1].split("'", 1)[0]
                    if item["mimeType"] != want_mime:
                        continue
                matches.append(item)
            return _FakeExec({"files": matches})

        def create(self, body: dict[str, Any], media_body: Any = None, **kwargs: Any) -> _FakeExec:
            self.seq += 1
            fid = f"fid_{self.seq}"
            raw = b""
            if media_body is not None:
                if isinstance(media_body, str):
                    p = Path(media_body)
                elif hasattr(media_body, "_filename"):
                    p = Path(media_body._filename)
                else:
                    p = Path(media_body.filename())
                raw = p.read_bytes()
            rec = {
                "id": fid,
                "name": body["name"],
                "parent": body["parents"][0],
                "mimeType": body.get("mimeType", "application/octet-stream"),
                "size": str(len(raw)),
                "md5Checksum": hashlib.md5(raw).hexdigest(),
                "modifiedTime": "2026-11-23T00:00:00Z",
            }
            self.items[fid] = rec
            self.blobs[fid] = raw
            return _FakeExec(rec)

        def get_media(self, fileId: str, **kwargs: Any) -> _FakeExec:
            return _FakeExec(self.blobs[fileId])

        def delete(self, fileId: str, **kwargs: Any) -> _FakeExec:
            self.items.pop(fileId, None)
            self.blobs.pop(fileId, None)
            return _FakeExec({})

    class _FakeDriveService:
        def __init__(self) -> None:
            self._files = _FakeFilesResource()

        def files(self) -> _FakeFilesResource:
            return self._files

    gdrive_api = GoogleDriveStorage(
        mode="api",
        folder_id="1IipfeWX_OVJS2KwaFL5wFlO-edxLoEGD",
        drive_service=_FakeDriveService(),
    )
    api_meta = gdrive_api.upload(sample, "releases/r1/shards/train/s0.tar")
    assert api_meta.size_bytes == sample.stat().st_size
    assert gdrive_api.exists("releases/r1/shards/train/s0.tar")
    dl_path = tmp_path / "dl_from_api.bin"
    gdrive_api.download("releases/r1/shards/train/s0.tar", dl_path, expected_sha256=api_meta.sha256)
    assert dl_path.read_bytes() == sample.read_bytes()
    assert gdrive_api.list_objects("releases/r1") == ["releases/r1/shards/train/s0.tar"]


def test_benchmark_profiler_runs_cleanly(tmp_path: Path) -> None:
    source_root = tmp_path / "bench_src"
    storage_root = tmp_path / "bench_storage"
    work_root = tmp_path / "bench_work"
    _build_tiny_genimage_fixture(source_root)
    rel_small_yaml, _, _ = _write_test_yaml_suite(tmp_path, source_root, storage_root, work_root)
    cfg = load_release_config(rel_small_yaml)
    report = run_pipeline_benchmark(cfg, cache_dir=tmp_path / "bench_cache", split="train")
    assert report["inventory"]["records"] == 72
    assert report["verification"]["passed"] is True
    assert report["shard_dataloader_read"]["samples_read"] == 8


@pytest.mark.skipif(
    not os.environ.get("PROOFLENS_REAL_GENIMAGE_ROOT"),
    reason="Opt-in real GenImage integration test; set PROOFLENS_REAL_GENIMAGE_ROOT on Colab to execute.",
)
def test_real_genimage_opt_in(tmp_path: Path) -> None:
    """Opt-in integration test verifying real Full GenImage inventory & dry-run feasibility on Colab."""
    real_root = Path(os.environ["PROOFLENS_REAL_GENIMAGE_ROOT"])
    assert real_root.exists(), f"PROOFLENS_REAL_GENIMAGE_ROOT not found: {real_root}"
    src_yaml = tmp_path / "real_src.yaml"
    src_yaml.write_text(
        yaml.safe_dump(
            {
                "source": {
                    "source_id": "genimage_real_opt_in",
                    "source_type": "directory" if real_root.is_dir() else "archive",
                    "source_revision": "colab-opt-in",
                    "location": str(real_root),
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "inventory",
                "--source-config",
                str(src_yaml),
                "--output",
                str(tmp_path / "inv.parquet"),
            ]
        )
        == 0
    )
