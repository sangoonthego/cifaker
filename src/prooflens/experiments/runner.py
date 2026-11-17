"""Experiment configuration validation and runner for ProofLens (Member 3 & Member 4).

Validates YAML experiment configurations:
- Rejects missing required sections and fields
- Rejects unsupported model names
- Validates numeric parameters (`seed`, `batch_size`, `max_batches`, `num_classes`, `image_size`)
- Supports overriding dataset and output paths
- Executes local CPU smoke runs cleanly
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from prooflens.data.dataset import ProofLensDataset, create_dataloader
from prooflens.data.scanner import scan_dataset_directory
from prooflens.evaluation.metrics import compute_binary_metrics
from prooflens.models.factory import ALL_SUPPORTED_MODELS, create_model
from prooflens.training.trainer import (
    SmokeTrainResult,
    generate_predictions,
    run_smoke_training_loop,
)

REQUIRED_CONFIG_SECTIONS: tuple[str, ...] = (
    "experiment",
    "dataset",
    "model",
    "training",
    "output",
)


@dataclass(frozen=True)
class ValidatedExperimentConfig:
    """Typed, validated experiment configuration."""

    experiment_name: str
    seed: int
    environment: str
    dataset_root: Path
    manifest_path: Path
    image_size: int
    model_name: str
    pretrained: bool
    num_classes: int
    device: str
    batch_size: int
    max_batches: int
    learning_rate: float
    output_dir: Path


def load_and_validate_experiment_config(
    config_path: Path | str,
    dataset_root_override: Path | str | None = None,
    manifest_override: Path | str | None = None,
    output_dir_override: Path | str | None = None,
) -> ValidatedExperimentConfig:
    """Load a YAML experiment config and validate all required fields and types."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")

    return validate_experiment_config_dict(
        raw,
        dataset_root_override=dataset_root_override,
        manifest_override=manifest_override,
        output_dir_override=output_dir_override,
    )


def validate_experiment_config_dict(
    raw: dict[str, Any],
    dataset_root_override: Path | str | None = None,
    manifest_override: Path | str | None = None,
    output_dir_override: Path | str | None = None,
) -> ValidatedExperimentConfig:
    """Validate an in-memory experiment configuration dictionary."""
    for sec in REQUIRED_CONFIG_SECTIONS:
        if sec not in raw or not isinstance(raw[sec], dict):
            raise ValueError(f"Missing or invalid required configuration section: '{sec}'")

    exp_sec = raw["experiment"]
    ds_sec = raw["dataset"]
    mod_sec = raw["model"]
    trn_sec = raw["training"]
    out_sec = raw["output"]

    exp_name = str(exp_sec.get("name", "")).strip()
    if not exp_name:
        raise ValueError("Missing required field 'experiment.name'.")

    if (
        "seed" not in exp_sec
        or not isinstance(exp_sec["seed"], int)
        or isinstance(exp_sec["seed"], bool)
    ):
        raise ValueError("Field 'experiment.seed' must be an integer.")
    seed = int(exp_sec["seed"])
    environment = str(exp_sec.get("environment", "local")).strip().lower()
    if environment not in {"local", "kaggle"}:
        raise ValueError(f"Unsupported 'experiment.environment': {environment}")

    ds_root_str = str(dataset_root_override or ds_sec.get("root", "")).strip()
    manifest_str = str(manifest_override or ds_sec.get("manifest", "")).strip()
    if not ds_root_str:
        raise ValueError("Missing required field 'dataset.root'.")
    if not manifest_str:
        raise ValueError("Missing required field 'dataset.manifest'.")

    image_size = int(ds_sec.get("image_size", 64))
    if image_size <= 0:
        raise ValueError(f"'dataset.image_size' must be positive, got {image_size}")

    model_name = str(mod_sec.get("name", "")).strip().lower()
    if not model_name:
        raise ValueError("Missing required field 'model.name'.")
    if model_name not in ALL_SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {ALL_SUPPORTED_MODELS}"
        )

    pretrained = bool(mod_sec.get("pretrained", False))
    num_classes = int(mod_sec.get("num_classes", 1))
    if num_classes != 1:
        raise ValueError(
            f"ProofLens binary classification contract requires 'model.num_classes == 1', got {num_classes}."
        )

    device = str(trn_sec.get("device", "cpu")).strip().lower()
    if (
        "batch_size" not in trn_sec
        or not isinstance(trn_sec["batch_size"], int)
        or trn_sec["batch_size"] <= 0
    ):
        raise ValueError("Field 'training.batch_size' must be a positive integer.")
    if (
        "max_batches" not in trn_sec
        or not isinstance(trn_sec["max_batches"], int)
        or trn_sec["max_batches"] <= 0
    ):
        raise ValueError("Field 'training.max_batches' must be a positive integer.")

    lr = float(trn_sec.get("learning_rate", 1e-3))
    if lr <= 0.0:
        raise ValueError(f"'training.learning_rate' must be positive, got {lr}")

    out_dir_str = str(output_dir_override or out_sec.get("directory", "")).strip()
    if not out_dir_str:
        raise ValueError("Missing required field 'output.directory'.")

    return ValidatedExperimentConfig(
        experiment_name=exp_name,
        seed=seed,
        environment=environment,
        dataset_root=Path(ds_root_str),
        manifest_path=Path(manifest_str),
        image_size=image_size,
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        device=device,
        batch_size=int(trn_sec["batch_size"]),
        max_batches=int(trn_sec["max_batches"]),
        learning_rate=lr,
        output_dir=Path(out_dir_str),
    )


def execute_smoke_experiment(config: ValidatedExperimentConfig) -> dict[str, Any]:
    """Run a complete CPU smoke experiment from a validated configuration."""
    if config.manifest_path.is_file():
        manifest_source: Any = config.manifest_path
    else:
        manifest_source = scan_dataset_directory(config.dataset_root)

    dataset = ProofLensDataset(
        root=config.dataset_root,
        manifest=manifest_source,
        image_size=config.image_size,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    model = create_model(
        name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.output_dir / "checkpoint_smoke.pt"

    smoke_res: SmokeTrainResult = run_smoke_training_loop(
        model=model,
        dataloader=dataloader,
        checkpoint_path=ckpt_path,
        model_name=config.model_name,
        device=config.device,
        learning_rate=config.learning_rate,
        max_batches=config.max_batches,
        seed=config.seed,
    )

    eval_loader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    preds_df = generate_predictions(
        model=model,
        dataloader=eval_loader,
        model_name=config.model_name,
        run_id=config.experiment_name,
        device=config.device,
    )
    preds_path = config.output_dir / "predictions_smoke.csv"
    preds_df.to_csv(preds_path, index=False)

    metrics = compute_binary_metrics(preds_df)
    summary_path = config.output_dir / "metrics_smoke.json"
    summary_payload = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "batches_executed": smoke_res.batches_executed,
        "initial_loss": smoke_res.initial_loss,
        "final_loss": smoke_res.final_loss,
        "gradients_verified": smoke_res.gradients_verified,
        "checkpoint_path": str(ckpt_path),
        "predictions_path": str(preds_path),
        "metrics": metrics,
        "disclaimer": (
            "Synthetic fixture smoke run for engineering verification only. "
            "Not a scientific or benchmark detection result."
        ),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload
