"""Unit tests for ProofLens dataset manifest contract, scanner, and PyTorch Dataset (Member 1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from prooflens.data.dataset import ProofLensDataset, create_dataloader
from prooflens.data.manifest import (
    LABEL_FAKE,
    LABEL_REAL,
    load_manifest_csv,
    save_manifest_csv,
    validate_manifest_dataframe,
)
from prooflens.data.scanner import parse_relative_image_path, scan_dataset_directory
from prooflens.data.validation import summarize_manifest_audit, verify_image_file


def test_parse_relative_image_path_standard_and_shallow() -> None:
    parsed = parse_relative_image_path("imagenet_ai_0419_biggan/train/ai/sample_01.png")
    assert parsed.generator == "imagenet_ai_0419_biggan"
    assert parsed.source_split == "train"
    assert parsed.label == LABEL_FAKE

    parsed_mapped = parse_relative_image_path(
        "imagenet_ai_0419_biggan/val/nature/sample_02.jpg",
        generator_mapping={"imagenet_ai_0419_biggan": "biggan"},
    )
    assert parsed_mapped.generator == "biggan"
    assert parsed_mapped.source_split == "val"
    assert parsed_mapped.label == LABEL_REAL

    parsed_no_gen = parse_relative_image_path("test/fake/sample_03.png")
    assert parsed_no_gen.generator == "unknown"
    assert parsed_no_gen.source_split == "test"
    assert parsed_no_gen.label == LABEL_FAKE


def test_parse_relative_image_path_rejects_ambiguous_folders() -> None:
    with pytest.raises(ValueError, match="Ambiguous or unmapped class folder"):
        parse_relative_image_path("biggan/train/mystery_class/img.png")

    with pytest.raises(ValueError, match="Ambiguous split hierarchy"):
        parse_relative_image_path("biggan/unknown_split/ai/img.png")

    with pytest.raises(ValueError, match="Ambiguous relative path"):
        parse_relative_image_path("ai/img.png")


def test_manifest_validation_rejects_invalid_rows() -> None:
    valid_df = pd.DataFrame(
        [
            {
                "image_id": "img_1",
                "relative_path": "gen_a/train/nature/01.jpg",
                "generator": "gen_a",
                "label": 0,
                "source_split": "train",
            },
            {
                "image_id": "img_2",
                "relative_path": "gen_a/train/ai/02.png",
                "generator": "gen_a",
                "label": 1,
                "source_split": "train",
            },
        ]
    )
    checked = validate_manifest_dataframe(valid_df)
    assert len(checked) == 2

    # Missing required column
    with pytest.raises(ValueError, match="missing required columns"):
        validate_manifest_dataframe(valid_df.drop(columns=["generator"]))

    # Invalid binary label (e.g. 2 or -1)
    bad_label = valid_df.copy()
    bad_label.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="invalid binary labels"):
        validate_manifest_dataframe(bad_label)

    # Duplicate image_id
    dup_id = valid_df.copy()
    dup_id.loc[1, "image_id"] = "img_1"
    with pytest.raises(ValueError, match="duplicate image_id"):
        validate_manifest_dataframe(dup_id)

    # Absolute path in relative_path
    abs_path = valid_df.copy()
    abs_path.loc[0, "relative_path"] = "/abs/path/img.jpg"
    with pytest.raises(ValueError, match="must be relative"):
        validate_manifest_dataframe(abs_path)

    # Unknown generator rejected when allow_unknown_generator=False
    unk_gen = valid_df.copy()
    unk_gen.loc[0, "generator"] = "unknown"
    with pytest.raises(ValueError, match="'unknown' generator"):
        validate_manifest_dataframe(unk_gen, allow_unknown_generator=False)


def test_scanner_corrupted_images_and_deterministic_loading(tmp_path: Path) -> None:
    # Create valid and corrupted fixture files
    real_dir = tmp_path / "mock_gen" / "train" / "nature"
    fake_dir = tmp_path / "mock_gen" / "train" / "ai"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    img1_path = real_dir / "valid_real.jpg"
    Image.new("RGB", (32, 32), color=(10, 120, 200)).save(img1_path, format="JPEG")

    img2_path = fake_dir / "valid_fake.png"
    Image.new("RGB", (32, 32), color=(200, 30, 50)).save(img2_path, format="PNG")

    corrupt_path = fake_dir / "corrupt_fake.png"
    corrupt_path.write_bytes(b"not_a_valid_png_stream")

    res_corrupt = verify_image_file(corrupt_path)
    assert not res_corrupt.is_valid
    assert res_corrupt.decode_status.startswith("CORRUPTED:")

    # Scan excluding corrupted
    df_clean = scan_dataset_directory(tmp_path, include_corrupted=False)
    assert len(df_clean) == 2

    # Scan including corrupted for audit
    df_audit = scan_dataset_directory(tmp_path, include_corrupted=True)
    assert len(df_audit) == 3
    summary = summarize_manifest_audit(df_audit)
    assert summary["total_files"] == 3
    assert summary["valid_images"] == 2
    assert summary["invalid_images"] == 1

    # Save & reload manifest CSV
    csv_path = tmp_path / "manifest.csv"
    save_manifest_csv(df_clean, csv_path)
    reloaded = load_manifest_csv(csv_path)
    assert list(reloaded["image_id"]) == list(df_clean["image_id"])

    # PyTorch Dataset & DataLoader deterministic loading
    ds = ProofLensDataset(root=tmp_path, manifest=csv_path, image_size=32)
    assert len(ds) == 2
    loader_a = create_dataloader(ds, batch_size=2, shuffle=False, seed=99)
    loader_b = create_dataloader(ds, batch_size=2, shuffle=False, seed=99)

    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    assert batch_a["image"].shape == (2, 3, 32, 32)
    assert (batch_a["image"] == batch_b["image"]).all()
    assert batch_a["image_id"] == batch_b["image_id"]
