"""Model interface contract and torchvision factory for ProofLens (Member 2).

Supports the four planned research architectures:
- `resnet50`
- `densenet121`
- `mobilenetv3_large`
- `convnext_tiny`
Plus a lightweight test-only CNN (`tiny_cnn`) for fast CPU smoke testing.

Contract:
- `create_model(name, pretrained=False, num_classes=1)`
- Forward pass outputs raw logits of shape `(batch_size, 1)` without sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_RESEARCH_MODELS: tuple[str, ...] = (
    "resnet50",
    "densenet121",
    "mobilenetv3_large",
    "convnext_tiny",
)

ALL_SUPPORTED_MODELS: tuple[str, ...] = SUPPORTED_RESEARCH_MODELS + ("tiny_cnn",)


class TinySmokeCNN(nn.Module):
    """Lightweight test-only CNN for fast CPU smoke testing of the training loop."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        return self.classifier(flat)


def create_model(
    name: str,
    pretrained: bool = False,
    num_classes: int = 1,
) -> nn.Module:
    """Instantiate a binary classification CNN adhering to the ProofLens logit contract.

    Args:
        name: Architecture identifier (`resnet50`, `densenet121`, `mobilenetv3_large`,
            `convnext_tiny`, or test-only `tiny_cnn`).
        pretrained: Whether to load ImageNet pretrained weights. Always pass `False`
            in local unit tests and CPU smoke tests.
        num_classes: Number of output logits. Must be `1` for binary classification
            with `BCEWithLogitsLoss`.

    Returns:
        PyTorch `nn.Module` producing raw logits of shape `(batch_size, num_classes)`.
    """
    norm_name = name.strip().lower()
    if norm_name not in ALL_SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model name '{name}'. Supported models: {ALL_SUPPORTED_MODELS}"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if norm_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
        return net

    if norm_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        net = models.densenet121(weights=weights)
        in_features = net.classifier.in_features
        net.classifier = nn.Linear(in_features, num_classes)
        return net

    if norm_name == "mobilenetv3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
        return net

    if norm_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
        return net

    return TinySmokeCNN(num_classes=num_classes)
