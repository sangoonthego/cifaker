"""Image integrity validation and dataset audit summary helpers for ProofLens."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

DEFAULT_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
)


@dataclass(frozen=True)
class ImageIntegrityResult:
    """Result of inspecting and decoding a single image file."""

    width: int | None
    height: int | None
    image_format: str
    file_size_bytes: int
    sha256: str
    decode_status: str
    is_valid: bool


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image_file(
    file_path: Path,
    supported_extensions: set[str] | frozenset[str] = DEFAULT_SUPPORTED_EXTENSIONS,
) -> ImageIntegrityResult:
    """Verify an image file's readability, dimensions, format, and SHA-256 hash."""
    if not file_path.is_file():
        return ImageIntegrityResult(
            width=None,
            height=None,
            image_format="MISSING",
            file_size_bytes=0,
            sha256="",
            decode_status="FILE_NOT_FOUND",
            is_valid=False,
        )

    file_size = file_path.stat().st_size
    sha256_hex = compute_file_sha256(file_path)
    ext = file_path.suffix.lower()

    if file_size == 0:
        return ImageIntegrityResult(
            width=None,
            height=None,
            image_format="EMPTY",
            file_size_bytes=0,
            sha256=sha256_hex,
            decode_status="EMPTY_FILE",
            is_valid=False,
        )

    if ext not in supported_extensions:
        return ImageIntegrityResult(
            width=None,
            height=None,
            image_format="UNSUPPORTED",
            file_size_bytes=file_size,
            sha256=sha256_hex,
            decode_status=f"UNSUPPORTED_EXTENSION:{ext}",
            is_valid=False,
        )

    try:
        with Image.open(file_path) as img:
            fmt = (img.format or "UNKNOWN").upper()
            width, height = img.size
            img.load()
        if width <= 0 or height <= 0:
            return ImageIntegrityResult(
                width=width,
                height=height,
                image_format=fmt,
                file_size_bytes=file_size,
                sha256=sha256_hex,
                decode_status="INVALID_DIMENSIONS",
                is_valid=False,
            )
        return ImageIntegrityResult(
            width=width,
            height=height,
            image_format=fmt,
            file_size_bytes=file_size,
            sha256=sha256_hex,
            decode_status="OK",
            is_valid=True,
        )
    except Exception as exc:
        return ImageIntegrityResult(
            width=None,
            height=None,
            image_format="CORRUPTED",
            file_size_bytes=file_size,
            sha256=sha256_hex,
            decode_status=f"CORRUPTED:{type(exc).__name__}",
            is_valid=False,
        )


def summarize_manifest_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a structured audit summary from a scanned or loaded manifest DataFrame."""
    total_rows = int(len(df))
    if total_rows == 0:
        return {
            "total_files": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "label_counts": {},
            "generator_counts": {},
            "split_counts": {},
            "duplicate_sha256_groups": 0,
        }

    has_decode = "decode_status" in df.columns
    valid_mask = (df["decode_status"] == "OK") if has_decode else pd.Series(True, index=df.index)
    valid_count = int(valid_mask.sum())
    invalid_count = total_rows - valid_count

    label_counts = {
        str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()
    }
    generator_counts = {
        str(k): int(v) for k, v in df["generator"].value_counts().sort_index().to_dict().items()
    }
    split_counts = {
        str(k): int(v) for k, v in df["source_split"].value_counts().sort_index().to_dict().items()
    }

    dup_groups = 0
    cross_split_dups = 0
    if "sha256" in df.columns:
        non_empty = df[df["sha256"].notna() & (df["sha256"].astype(str) != "")]
        counts = non_empty["sha256"].value_counts()
        dup_hashes = counts[counts > 1].index
        dup_groups = int(len(dup_hashes))
        for sha in dup_hashes:
            splits = non_empty.loc[non_empty["sha256"] == sha, "source_split"].nunique()
            if splits > 1:
                cross_split_dups += 1

    format_counts: dict[str, int] = {}
    if "image_format" in df.columns:
        format_counts = {
            str(k): int(v) for k, v in Counter(df["image_format"].dropna().astype(str)).items()
        }

    return {
        "total_files": total_rows,
        "valid_images": valid_count,
        "invalid_images": invalid_count,
        "label_counts": label_counts,
        "generator_counts": generator_counts,
        "split_counts": split_counts,
        "image_formats": format_counts,
        "duplicate_sha256_groups": dup_groups,
        "cross_split_duplicate_groups": cross_split_dups,
    }
