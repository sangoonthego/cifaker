"""Minimal CPU smoke trainer and deterministic seeding for ProofLens (Member 2)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prooflens.training.checkpoint import save_checkpoint


def set_deterministic_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible CPU/GPU execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class SmokeTrainResult:
    """Summary returned by a CPU smoke training run."""

    model_name: str
    batches_executed: int
    initial_loss: float
    final_loss: float
    loss_history: list[float]
    gradients_verified: bool
    checkpoint_path: Path


def run_smoke_training_loop(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    checkpoint_path: Path | str,
    model_name: str = "resnet50",
    device: str | torch.device = "cpu",
    learning_rate: float = 1e-3,
    max_batches: int = 2,
    seed: int = 42,
) -> SmokeTrainResult:
    """Execute a minimal CPU training loop to validate pipeline wiring.

    Pipeline verified:
        Dataset -> DataLoader -> Model -> Forward Pass -> BCEWithLogitsLoss
        -> Backpropagation -> Gradient Verification -> Optimizer Step -> Checkpoint
    """
    if max_batches <= 0:
        raise ValueError(f"max_batches must be positive, got {max_batches}")

    set_deterministic_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []
    gradients_verified = False
    batches_run = 0

    for batch in dataloader:
        if batches_run >= max_batches:
            break

        images = batch["image"].to(dev)
        labels = batch["label"].to(dev).view(-1, 1)

        optimizer.zero_grad()
        logits = model(images)

        if logits.shape != (images.shape[0], 1):
            raise RuntimeError(
                f"Model output contract violation: expected shape {(images.shape[0], 1)}, "
                f"got {tuple(logits.shape)}"
            )

        loss = criterion(logits, labels)
        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            raise RuntimeError(f"Non-finite training loss encountered: {loss_val}")

        loss.backward()

        # Verify that trainable parameters received finite, non-None gradients
        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        if not has_grad:
            raise RuntimeError("No finite gradients were computed during backpropagation.")
        gradients_verified = True

        optimizer.step()
        loss_history.append(loss_val)
        batches_run += 1

    if batches_run == 0:
        raise RuntimeError("DataLoader yielded 0 batches during smoke training.")

    saved_ckpt = save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        step=batches_run,
        loss=loss_history[-1],
        model_name=model_name,
        extra_metadata={"seed": seed, "loss_history": loss_history},
    )

    return SmokeTrainResult(
        model_name=model_name,
        batches_executed=batches_run,
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        loss_history=loss_history,
        gradients_verified=gradients_verified,
        checkpoint_path=saved_ckpt,
    )


@torch.no_grad()
def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    model_name: str,
    run_id: str = "local_smoke",
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Run inference over a DataLoader and produce a DataFrame conforming to the Prediction Contract.

    Applies `torch.sigmoid` to raw model logits in the evaluation/prediction layer.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    rows: list[dict[str, Any]] = []
    for batch in dataloader:
        images = batch["image"].to(dev)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise RuntimeError(f"Expected logits shape (B, 1), got {tuple(logits.shape)}")

        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        y_true_arr = batch["label"].cpu().numpy().astype(int)
        image_ids = batch["image_id"]
        generators = batch["generator"]

        for img_id, gen, yt, ys in zip(image_ids, generators, y_true_arr, scores, strict=True):
            score_float = float(ys)
            pred_int = 1 if score_float >= threshold else 0
            rows.append(
                {
                    "image_id": str(img_id),
                    "model": model_name,
                    "generator": str(gen),
                    "y_true": int(yt),
                    "y_score": score_float,
                    "y_pred": pred_int,
                    "run_id": run_id,
                }
            )

    return pd.DataFrame(rows)
