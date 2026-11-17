"""Model architecture package for ProofLens (Member 2)."""

from __future__ import annotations

from prooflens.models.factory import (
    ALL_SUPPORTED_MODELS,
    SUPPORTED_RESEARCH_MODELS,
    TinySmokeCNN,
    create_model,
)

__all__ = [
    "ALL_SUPPORTED_MODELS",
    "SUPPORTED_RESEARCH_MODELS",
    "TinySmokeCNN",
    "create_model",
]
