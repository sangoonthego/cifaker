"""PyTorch Dataset and DataLoader integration for ProofLens."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prooflens.data.manifest import load_manifest_csv, validate_manifest_dataframe

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_image_transforms(
    image_size: int = 64,
    normalize: bool = True,
) -> transforms.Compose:
    """Create configurable, deterministic basic image transforms for RGB tensors."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    steps: list[Callable[..., Any]] = [
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


class ProofLensDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset backed by a validated ProofLens manifest DataFrame or CSV."""

    def __init__(
        self,
        root: Path | str,
        manifest: pd.DataFrame | Path | str,
        split: str | None = None,
        generators: list[str] | set[str] | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_size: int = 64,
        allow_unknown_generator: bool = True,
    ) -> None:
        self.root = Path(root)
        if isinstance(manifest, (str, Path)):
            df = load_manifest_csv(manifest, allow_unknown_generator=allow_unknown_generator)
        else:
            df = validate_manifest_dataframe(
                manifest, allow_unknown_generator=allow_unknown_generator
            )

        # Filter out rows known to have failed decoding if decode_status column is present
        if "decode_status" in df.columns:
            df = df[df["decode_status"] == "OK"].copy()

        if split is not None:
            df = df[df["source_split"] == split].copy()

        if generators is not None:
            allowed = {str(g) for g in generators}
            df = df[df["generator"].isin(allowed)].copy()

        self.manifest = df.reset_index(drop=True)
        self.transform = (
            transform if transform is not None else build_image_transforms(image_size=image_size)
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def resolve_image_path(self, relative_path: str) -> Path:
        """Resolve a manifest `relative_path` safely against `self.root`."""
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Invalid relative path in manifest: {relative_path}")
        full_path = self.root / rel
        if not full_path.is_file():
            raise FileNotFoundError(f"Image file not found on disk: {full_path}")
        return full_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        img_path = self.resolve_image_path(str(row["relative_path"]))

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")

        tensor_img = self.transform(rgb_img)
        label_int = int(row["label"])

        return {
            "image": tensor_img,
            "label": torch.tensor(label_int, dtype=torch.float32),
            "image_id": str(row["image_id"]),
            "generator": str(row["generator"]),
            "source_split": str(row["source_split"]),
            "relative_path": str(row["relative_path"]),
        }


def create_dataloader(
    dataset: ProofLensDataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader[dict[str, Any]]:
    """Build a deterministic PyTorch DataLoader for CPU or GPU execution."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
