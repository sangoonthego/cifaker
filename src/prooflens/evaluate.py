"""Evaluation pipeline placeholder for ProofLens (Deferred to Phase 1+).

Planned evaluations for later phases:
- In-domain benchmark evaluation
- Cross-generator zero-shot generalization
- Perturbation and compression robustness
- Subgroup and failure-mode analysis

NOTE: No model evaluation is executed during Phase 0.
"""

from __future__ import annotations

from pathlib import Path


def run_evaluation(config_path: Path | str, checkpoint_path: Path | str) -> None:
    """Placeholder entry point for Phase 1+ model evaluation."""
    raise NotImplementedError(
        f"Model evaluation (config={config_path}, checkpoint={checkpoint_path}) "
        "is deferred to Phase 1+."
    )
