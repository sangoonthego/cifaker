"""Model architecture registry for ProofLens (Phase 1+ Placeholder).

Planned CNN architectures for AI-generated image detection:
1. ResNet-50 (`resnet50`)
2. DenseNet-121 (`densenet121`)
3. MobileNetV3-Large (`mobilenet_v3_large`)
4. ConvNeXt-Tiny (`convnext_tiny`)

NOTE: In accordance with Phase 0 scope boundaries, no neural network training or
model instantiation is executed during Phase 0.
"""

from __future__ import annotations

PLANNED_MODELS: tuple[str, ...] = (
    "resnet50",
    "densenet121",
    "mobilenet_v3_large",
    "convnext_tiny",
)


def build_model(architecture: str, num_classes: int = 2, pretrained: bool = True) -> object:
    """Placeholder factory for Phase 1+ CNN architectures."""
    if architecture not in PLANNED_MODELS:
        raise ValueError(f"Unsupported architecture '{architecture}'. Planned: {PLANNED_MODELS}")
    raise NotImplementedError(
        f"Model construction for '{architecture}' (num_classes={num_classes}, pretrained={pretrained}) "
        "is deferred to Phase 1. Phase 0 covers dataset acquisition and auditing only."
    )
