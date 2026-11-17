"""Training pipeline placeholder for ProofLens (Deferred to Phase 1+).

NOTE: Model training, fine-tuning, and GPU execution are strictly out of scope for Phase 0.
"""

from __future__ import annotations

from pathlib import Path


def run_training(config_path: Path | str) -> None:
    """Placeholder entry point for Phase 1+ model training."""
    raise NotImplementedError(
        f"Training execution (config={config_path}) is deferred to Phase 1. "
        "Complete and validate Phase 0 dataset engineering first."
    )
