"""Unit tests for canonical registry schema, YAML configs, source adapters, and duplicate detection."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from prooflens.data.config import (
    ConfigValidationError,
    SourceConfig,
    load_release_config,
    validate_protocol_config,
    validate_source_config,
)
from prooflens.data.processing.validator import validate_image_bytes
from prooflens.data.registry.duplicates import (
    assign_leakage_groups,
    find_perceptual_near_duplicate_candidates,
)
from prooflens.data.registry.inventory import build_source_inventory
from prooflens.data.registry.schema import (
    LABEL_FAKE,
    LABEL_REAL,
    CanonicalImageRecord,
    extract_imagenet_class_from_path,
    make_canonical_image_id,
    read_canonical_parquet,
    validate_canonical_dataframe,
    validate_safe_relative_path,
    write_canonical_parquet,
)
from prooflens.data.sources.archive import ArchiveSourceAdapter
from prooflens.data.sources.base import SourceAccessError
from prooflens.data.sources.genimage import DirectoryGenImageAdapter


def _save_rgb_image(path: Path, seed: int, fmt: str = "PNG", size: int = 32) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format=fmt)


def test_canonical_schema_provenance_and_parquet_roundtrip(tmp_path: Path) -> None:
    # REAL image must NOT have a generative model name in `generator`
    with pytest.raises(ValueError, match="REAL image"):
        CanonicalImageRecord(
            image_id="pl_real_bad",
            source_id="src1",
            source_revision="v1",
            source_path="biggan/train/nature/n01440764_1.jpg",
            source_split="train",
            label=LABEL_REAL,
            generator="biggan",
            source_generator_group="biggan",
        )

    # FAKE image must specify a generative model name
    with pytest.raises(ValueError, match="FAKE image"):
        CanonicalImageRecord(
            image_id="pl_fake_bad",
            source_id="src1",
            source_revision="v1",
            source_path="biggan/train/ai/000_biggan_00001.png",
            source_split="train",
            label=LABEL_FAKE,
            generator="none",
            source_generator_group="biggan",
        )

    # Unsafe paths rejected
    for bad_path in ("/etc/passwd", "C:/Windows/a.png", "biggan/../secret.png", ""):
        with pytest.raises(ValueError):
            validate_safe_relative_path(bad_path)

    # Stable provenance-derived IDs
    id_a = make_canonical_image_id("genimage", "rev1", "biggan/train/ai/000_biggan_00001.png")
    id_b = make_canonical_image_id("genimage", "rev1", "biggan\\train\\ai\\000_biggan_00001.png")
    assert id_a == id_b
    assert id_a.startswith("pl_")
    assert (
        extract_imagenet_class_from_path("biggan/train/nature/n01440764_10026.JPEG") == "n01440764"
    )
    assert extract_imagenet_class_from_path("biggan/train/ai/007_biggan_00093.png") == "cls_007"

    rec_real = CanonicalImageRecord(
        image_id="pl_0001",
        source_id="src1",
        source_revision="v1",
        source_path="biggan/train/nature/n01440764_1.jpg",
        source_split="train",
        label=LABEL_REAL,
        generator="none",
        source_generator_group="biggan",
        imagenet_class="n01440764",
    )
    rec_fake = CanonicalImageRecord(
        image_id="pl_0002",
        source_id="src1",
        source_revision="v1",
        source_path="biggan/train/ai/000_biggan_00001.png",
        source_split="train",
        label=LABEL_FAKE,
        generator="biggan",
        source_generator_group="biggan",
        imagenet_class="cls_000",
    )

    df = pd.DataFrame([rec_real.to_dict(), rec_fake.to_dict()])
    pq_path = tmp_path / "canonical.parquet"
    write_canonical_parquet(df, pq_path)
    loaded = read_canonical_parquet(pq_path)
    assert len(loaded) == 2
    assert list(loaded["generator"]) == ["none", "biggan"]

    # Reject duplicate canonical image_id
    dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate canonical image_id"):
        validate_canonical_dataframe(dup_df)


def test_yaml_configs_validate_and_reject_invalid_fields() -> None:
    # Verify shipped 120K / 200K / 500K release configs load and validate cleanly
    cfg120 = load_release_config("configs/releases/genimage_120k_v1.yaml")
    cfg200 = load_release_config("configs/releases/genimage_200k_v1.yaml")
    cfg500 = load_release_config("configs/releases/genimage_500k_v1.yaml")

    assert cfg120.release_id == "genimage-120k-v1"
    assert cfg120.protocol.splits["train"].total_quota == 80_000
    assert cfg120.protocol.splits["dev"].total_quota == 10_000
    assert cfg120.protocol.splits["seen_test"].total_quota == 10_000
    assert cfg120.protocol.splits["unseen_test"].total_quota == 20_000
    assert cfg200.parent_release_id == "genimage-120k-v1"
    assert cfg200.protocol.splits["train"].total_quota == 160_000
    assert cfg500.parent_release_id == "genimage-200k-v1"
    assert cfg500.protocol.splits["train"].total_quota == 460_000
    assert cfg120.storage.gdrive_folder_id == "1IipfeWX_OVJS2KwaFL5wFlO-edxLoEGD"
    assert len(cfg120.config_hash) == 64

    # Reject unknown critical field
    with pytest.raises(ConfigValidationError, match="Unknown configuration field"):
        validate_source_config(
            {
                "source_id": "s1",
                "source_type": "directory",
                "source_revision": "r1",
                "location": "data",
                "unexpected_key": 123,
            }
        )

    # Reject unseen generator pool in train split
    with pytest.raises(ConfigValidationError, match="exclusively from 'seen'"):
        validate_protocol_config(
            {
                "protocol_version": "bad_v1",
                "seen_generators": ["biggan"],
                "unseen_generators": ["midjourney"],
                "splits": {
                    "train": {
                        "allowed_source_splits": ["train"],
                        "generator_pool": "unseen",
                        "real_quota": 10,
                        "fake_quota": 10,
                    }
                },
            }
        )


def test_directory_and_archive_source_adapters(tmp_path: Path) -> None:
    # Build tiny synthetic directory source
    dir_root = tmp_path / "dir_src"
    _save_rgb_image(dir_root / "imagenet_ai_0419_biggan/train/nature/n01440764_01.jpg", 1, "JPEG")
    _save_rgb_image(dir_root / "imagenet_ai_0419_biggan/train/ai/000_biggan_00001.png", 2, "PNG")

    dir_cfg = SourceConfig(
        source_id="tiny_dir",
        source_type="directory",
        source_revision="v1",
        location=str(dir_root),
        generator_mapping={"imagenet_ai_0419_biggan": "biggan"},
    )
    dir_adapter = DirectoryGenImageAdapter(dir_cfg)
    assert dir_adapter.capabilities().metadata_only_enumeration is True
    assert dir_adapter.capabilities().random_access is True

    inv_res = build_source_inventory(
        dir_adapter,
        output_parquet=tmp_path / "inv.parquet",
        batch_size=1,
        validate_image_bytes=True,
    )
    assert inv_res.total_records == 2
    inv_df = read_canonical_parquet(inv_res.parquet_path)
    real_row = inv_df[inv_df["label"] == LABEL_REAL].iloc[0]
    fake_row = inv_df[inv_df["label"] == LABEL_FAKE].iloc[0]
    assert real_row["generator"] == "none"
    assert real_row["source_generator_group"] == "biggan"
    assert fake_row["generator"] == "biggan"

    # Build ZIP and TAR.GZ archive sources and verify shard-aware streaming
    arch_dir = tmp_path / "archives"
    arch_dir.mkdir()
    zip_path = arch_dir / "biggan.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(
            dir_root / "imagenet_ai_0419_biggan/train/nature/n01440764_01.jpg",
            arcname="train/nature/n01440764_01.jpg",
        )
    tgz_path = arch_dir / "midjourney.tar.gz"
    with tarfile.open(tgz_path, "w:gz") as tf:
        tf.add(
            dir_root / "imagenet_ai_0419_biggan/train/ai/000_biggan_00001.png",
            arcname="val/ai/000_mj_00001.png",
        )

    arch_cfg = SourceConfig(
        source_id="tiny_arch",
        source_type="archive",
        source_revision="v1",
        location=str(arch_dir),
    )
    arch_adapter = ArchiveSourceAdapter(arch_cfg)
    assert arch_adapter.capabilities().shard_level_access is True
    assert arch_adapter.capabilities().random_access is False  # mixed zip + tar.gz

    records = list(arch_adapter.iter_records())
    assert len(records) == 2

    # Sequential tar.gz must reject single-image random access
    tgz_rec = next(r for r in records if r.source_shard == "midjourney.tar.gz")
    with pytest.raises(SourceAccessError, match="Single-image random access is disabled"):
        arch_adapter.fetch_image_bytes(tgz_rec)

    # Shard-aware streaming reads all selected members cleanly
    streamed = list(arch_adapter.iter_selected_images(records))
    assert len(streamed) == 2
    assert all(payload is not None and err is None for _, payload, err in streamed)


def test_exact_pixel_and_lsh_perceptual_duplicate_detection(tmp_path: Path) -> None:
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p3 = tmp_path / "c.bmp"
    p4 = tmp_path / "d.png"

    _save_rgb_image(p1, seed=10, fmt="PNG", size=32)
    p2.write_bytes(p1.read_bytes())  # exact SHA-256 duplicate
    with Image.open(p1) as img:
        img.save(p3, format="BMP")  # identical RGB pixels, different file bytes/format
    _save_rgb_image(p4, seed=99, fmt="PNG", size=32)

    v1 = validate_image_bytes(
        p1.read_bytes(), ".png", compute_pixel_hash=True, compute_perceptual_hash=True
    )
    v2 = validate_image_bytes(
        p2.read_bytes(), ".png", compute_pixel_hash=True, compute_perceptual_hash=True
    )
    v3 = validate_image_bytes(
        p3.read_bytes(), ".bmp", compute_pixel_hash=True, compute_perceptual_hash=True
    )
    v4 = validate_image_bytes(
        p4.read_bytes(), ".png", compute_pixel_hash=True, compute_perceptual_hash=True
    )

    assert v1.sha256 == v2.sha256
    assert v1.sha256 != v3.sha256
    assert v1.pixel_hash == v3.pixel_hash
    assert v1.perceptual_hash == v3.perceptual_hash

    df = pd.DataFrame(
        [
            {
                "image_id": "pl_1",
                "source_split": "train",
                "label": 1,
                "sha256": v1.sha256,
                "pixel_hash": v1.pixel_hash,
                "perceptual_hash": v1.perceptual_hash,
                "leakage_group_id": "lg_1",
            },
            {
                "image_id": "pl_2",
                "source_split": "val",
                "label": 1,
                "sha256": v2.sha256,
                "pixel_hash": v2.pixel_hash,
                "perceptual_hash": v2.perceptual_hash,
                "leakage_group_id": "lg_2",
            },
            {
                "image_id": "pl_3",
                "source_split": "train",
                "label": 1,
                "sha256": v3.sha256,
                "pixel_hash": v3.pixel_hash,
                "perceptual_hash": v3.perceptual_hash,
                "leakage_group_id": "lg_3",
            },
            {
                "image_id": "pl_4",
                "source_split": "train",
                "label": 1,
                "sha256": v4.sha256,
                "pixel_hash": v4.pixel_hash,
                "perceptual_hash": v4.perceptual_hash,
                "leakage_group_id": "lg_4",
            },
        ]
    )

    updated, rep = assign_leakage_groups(
        df,
        audit_scope="unit_test_subset",
        include_pixel_hash=True,
        include_perceptual_in_leakage_groups=False,
    )
    assert rep.exact_sha256_duplicate_groups == 1
    assert rep.cross_source_split_sha256_groups == 1
    assert rep.pixel_hash_duplicate_groups == 1
    assert rep.perceptual_candidate_pairs >= 1
    assert rep.perceptual_included_in_leakage_groups is False
    # pl_1, pl_2 (exact sha256), and pl_3 (exact pixel_hash) share one leakage group; pl_4 is separate
    groups = dict(zip(updated["image_id"], updated["leakage_group_id"], strict=True))
    assert groups["pl_1"] == groups["pl_2"] == groups["pl_3"]
    assert groups["pl_4"] != groups["pl_1"]

    # Band LSH finds Hamming distance 1 pair cleanly without O(N^2) scan
    cands = find_perceptual_near_duplicate_candidates(
        ["a", "b", "c"],
        ["0000111122223333", "0000111122223332", "ffff0000aaaa5555"],
        max_hamming_distance=2,
    )
    assert len(cands) == 1
    assert cands[0]["image_id_a"] == "a"
    assert cands[0]["image_id_b"] == "b"
    assert cands[0]["hamming_distance"] == 1
