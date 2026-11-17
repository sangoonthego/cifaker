"""Unit tests for CPU smoke training, config validation, and checkpoint round-trip (Member 2 & 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from scripts.generate_fixtures import generate_synthetic_fixtures

from prooflens.data.dataset import ProofLensDataset, create_dataloader
from prooflens.experiments.runner import validate_experiment_config_dict
from prooflens.models.factory import create_model
from prooflens.training.checkpoint import load_checkpoint
from prooflens.training.trainer import generate_predictions, run_smoke_training_loop


def test_cpu_smoke_training_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    manifest_csv = tmp_path / "manifests" / "fixtures.csv"
    generate_synthetic_fixtures(
        root_dir=fixtures_root,
        manifest_path=manifest_csv,
        image_size=64,
        seed=42,
    )

    dataset = ProofLensDataset(root=fixtures_root, manifest=manifest_csv, image_size=64)
    loader = create_dataloader(dataset, batch_size=2, shuffle=True, seed=42)

    model = create_model(name="tiny_cnn", pretrained=False, num_classes=1)
    ckpt_path = tmp_path / "checkpoints" / "smoke.pt"

    result = run_smoke_training_loop(
        model=model,
        dataloader=loader,
        checkpoint_path=ckpt_path,
        model_name="tiny_cnn",
        device="cpu",
        max_batches=2,
        seed=42,
    )

    assert result.batches_executed == 2
    assert result.gradients_verified is True
    assert ckpt_path.is_file()

    # Verify checkpoint loading into a fresh model instance produces identical outputs
    restored_model = create_model(name="tiny_cnn", pretrained=False, num_classes=1)
    loaded_payload = load_checkpoint(ckpt_path, model=restored_model, map_location="cpu")
    assert loaded_payload["model_name"] == "tiny_cnn"
    assert loaded_payload["step"] == 2

    model.eval()
    restored_model.eval()
    test_x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out_orig = model(test_x)
        out_restored = restored_model(test_x)
    assert torch.allclose(out_orig, out_restored)

    # Verify prediction generation contract
    preds_df = generate_predictions(
        model=restored_model,
        dataloader=loader,
        model_name="tiny_cnn",
        run_id="smoke_unit_test",
        device="cpu",
    )
    assert len(preds_df) == len(dataset)
    assert set(preds_df.columns) == {
        "image_id",
        "model",
        "generator",
        "y_true",
        "y_score",
        "y_pred",
        "run_id",
    }


def test_experiment_config_validation_rejects_invalid_configs() -> None:
    base_cfg = {
        "experiment": {"name": "smoke", "seed": 42},
        "dataset": {"root": "data/fixtures", "manifest": "data/manifests/fixtures.csv"},
        "model": {"name": "resnet50", "pretrained": False, "num_classes": 1},
        "training": {"device": "cpu", "batch_size": 2, "max_batches": 2},
        "output": {"directory": "artifacts/smoke"},
    }
    validated = validate_experiment_config_dict(base_cfg)
    assert validated.model_name == "resnet50"

    # Unsupported model name
    bad_model = {**base_cfg, "model": {"name": "alexnet", "pretrained": False, "num_classes": 1}}
    with pytest.raises(ValueError, match="Unsupported model"):
        validate_experiment_config_dict(bad_model)

    # Invalid num_classes != 1
    bad_classes = {**base_cfg, "model": {"name": "resnet50", "pretrained": False, "num_classes": 2}}
    with pytest.raises(ValueError, match="num_classes == 1"):
        validate_experiment_config_dict(bad_classes)

    # Invalid batch_size <= 0
    bad_batch = {**base_cfg, "training": {"device": "cpu", "batch_size": 0, "max_batches": 2}}
    with pytest.raises(ValueError, match="batch_size"):
        validate_experiment_config_dict(bad_batch)
