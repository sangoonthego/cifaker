"""Deterministic synthetic test fixture generator for ProofLens (Phase 0A).

Generates 24 small RGB synthetic images (64x64) across two mock generators
(`mock_gan`, `mock_diffusion`), three source splits (`train`, `val`, `test`),
and both binary classes (`nature` -> REAL=0, `ai` -> FAKE=1), using five distinct
visual patterns:
1. Uniform color
2. Deterministic random noise
3. Horizontal/vertical gradients
4. Checkerboard grid
5. Simple geometric shapes (circle/rectangle)

Also writes the corresponding validated manifest CSV (`data/manifests/fixtures.csv`).

IMPORTANT: These synthetic fixtures are strictly for local engineering validation
and CI testing. Never interpret metrics on these fixtures as scientific results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prooflens.data.manifest import save_manifest_csv
from prooflens.data.scanner import scan_dataset_directory


def _make_pattern_image(pattern_idx: int, rng: np.random.Generator, size: int = 64) -> Image.Image:
    """Create a single deterministic 64x64 RGB synthetic image according to `pattern_idx`."""
    kind = pattern_idx % 5

    if kind == 0:
        color = tuple(int(c) for c in rng.integers(20, 235, size=3))
        return Image.new("RGB", (size, size), color=color)

    if kind == 1:
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    if kind == 2:
        x = np.linspace(0, 255, size, dtype=np.uint8)
        y = np.linspace(255, 0, size, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        b_chan = np.full((size, size), int(rng.integers(40, 200)), dtype=np.uint8)
        arr = np.stack([xv, yv, b_chan], axis=-1)
        return Image.fromarray(arr)

    if kind == 3:
        tile = 8
        coords = np.indices((size, size))
        mask = ((coords[0] // tile) + (coords[1] // tile)) % 2 == 0
        c1 = rng.integers(10, 100, size=3, dtype=np.uint8)
        c2 = rng.integers(150, 245, size=3, dtype=np.uint8)
        arr = np.where(mask[..., None], c1, c2).astype(np.uint8)
        return Image.fromarray(arr)

    bg = tuple(int(c) for c in rng.integers(10, 80, size=3))
    fg1 = tuple(int(c) for c in rng.integers(120, 250, size=3))
    fg2 = tuple(int(c) for c in rng.integers(100, 240, size=3))
    img = Image.new("RGB", (size, size), color=bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size - 9, size - 9], outline=fg1, width=3)
    draw.ellipse([16, 16, size - 17, size - 17], fill=fg2)
    return img


def generate_synthetic_fixtures(
    root_dir: Path | str = Path("data/fixtures"),
    manifest_path: Path | str = Path("data/manifests/fixtures.csv"),
    image_size: int = 64,
    seed: int = 42,
) -> Path:
    """Generate 24 deterministic synthetic fixture images and save their validated manifest."""
    root = Path(root_dir)
    rng = np.random.default_rng(seed)

    generators = ("mock_gan", "mock_diffusion")
    split_counts = (("train", 3), ("val", 2), ("test", 1))
    classes = (("nature", ".jpg"), ("ai", ".png"))

    idx = 0
    for gen in generators:
        for split_name, count_per_class in split_counts:
            for class_folder, ext in classes:
                target_folder = root / gen / split_name / class_folder
                target_folder.mkdir(parents=True, exist_ok=True)
                for sample_num in range(count_per_class):
                    img = _make_pattern_image(idx, rng, size=image_size)
                    fname = f"{gen}_{split_name}_{class_folder}_{sample_num:02d}{ext}"
                    file_path = target_folder / fname
                    if ext == ".jpg":
                        img.save(file_path, format="JPEG", quality=92)
                    else:
                        img.save(file_path, format="PNG")
                    idx += 1

    manifest_df = scan_dataset_directory(root)
    saved_manifest = save_manifest_csv(manifest_df, manifest_path)
    return saved_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic test fixtures and manifest for ProofLens."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/fixtures"),
        help="Output directory for synthetic fixture images (default: data/fixtures).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fixtures.csv"),
        help="Output path for fixture manifest CSV (default: data/manifests/fixtures.csv).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Width and height of synthetic RGB images in pixels (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    args = parser.parse_args(argv)

    manifest_out = generate_synthetic_fixtures(
        root_dir=args.root,
        manifest_path=args.manifest,
        image_size=args.size,
        seed=args.seed,
    )
    print(f"[ProofLens] Generated synthetic fixtures in: {args.root}")
    print(f"[ProofLens] Saved validated fixture manifest to: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
