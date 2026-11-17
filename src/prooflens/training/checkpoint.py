"""Checkpoint serialization and loading utilities for ProofLens (Member 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
    loss: float | None = None,
    model_name: str = "unknown",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialize model weights, optimizer state, and training metadata to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "epoch": int(epoch),
        "step": int(step),
        "loss": float(loss) if loss is not None else None,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metadata": extra_metadata or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: Path | str,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a ProofLens checkpoint from disk and optionally restore model/optimizer state."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    payload: dict[str, Any] = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format (missing 'model_state_dict'): {ckpt_path}")

    if model is not None:
        model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload
