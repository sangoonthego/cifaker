"""Unit tests for ProofLens model factory and output contract (Member 2)."""

from __future__ import annotations

import pytest
import torch

from prooflens.models.factory import SUPPORTED_RESEARCH_MODELS, create_model


@pytest.mark.parametrize("model_name", SUPPORTED_RESEARCH_MODELS)
def test_all_four_models_adhere_to_single_logit_contract(model_name: str) -> None:
    """Verify all four torchvision architectures instantiate with pretrained=False and output (B, 1) logits."""
    batch_size = 2
    # 64x64 is >= 32x32 minimum spatial size for ResNet/DenseNet/MobileNetV3/ConvNeXt downsampling stages
    dummy_input = torch.randn(batch_size, 3, 64, 64)

    model = create_model(name=model_name, pretrained=False, num_classes=1)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_input)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()


def test_create_model_rejects_unsupported_name() -> None:
    with pytest.raises(ValueError, match="Unsupported model name"):
        create_model(name="vgg19_nonexistent", pretrained=False, num_classes=1)
