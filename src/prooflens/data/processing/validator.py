"""Single-pass raw image bytes integrity validation, hashing, and perceptual dHash.

Validates original image bytes in memory without re-encoding, resizing, or modifying
the payload:
- Exact SHA-256 of raw bytes
- Full Pillow pixel decoding check (`ImageFile.LOAD_TRUNCATED_IMAGES = False`)
- Format & positive dimension check
- Optional decoded RGB pixel-buffer SHA-256 (`pixel_hash`)
- Optional 64-bit difference hash (`perceptual_hash`, dHash) computed in the same decode pass
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
)


@dataclass(frozen=True)
class ByteValidationResult:
    """Validation result for raw image bytes."""

    is_valid: bool
    decode_status: str
    width: int | None
    height: int | None
    image_format: str | None
    file_size_bytes: int
    sha256: str
    pixel_hash: str | None = None
    perceptual_hash: str | None = None


def compute_dhash64(image: Image.Image) -> str:
    """Compute a 64-bit horizontal difference hash (dHash) returned as a 16-char hex string."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for val in diff.ravel():
        bits = (bits << 1) | int(bool(val))
    return f"{bits:016x}"


def validate_image_bytes(
    raw_bytes: bytes,
    file_extension: str = "",
    supported_extensions: set[str] | frozenset[str] = SUPPORTED_IMAGE_EXTENSIONS,
    compute_pixel_hash: bool = False,
    compute_perceptual_hash: bool = True,
) -> ByteValidationResult:
    """Validate raw image bytes and extract metadata/hashes in a single decode pass."""
    size = len(raw_bytes)
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    ext = file_extension.lower().strip()

    if size == 0:
        return ByteValidationResult(
            is_valid=False,
            decode_status="EMPTY_FILE",
            width=None,
            height=None,
            image_format=None,
            file_size_bytes=0,
            sha256=sha256_hex,
        )

    if ext and ext not in supported_extensions:
        return ByteValidationResult(
            is_valid=False,
            decode_status=f"UNSUPPORTED_EXTENSION:{ext}",
            width=None,
            height=None,
            image_format="UNSUPPORTED",
            file_size_bytes=size,
            sha256=sha256_hex,
        )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            fmt = (img.format or "UNKNOWN").upper()
            width, height = img.size
            # Decode full pixel data to catch truncated streams
            img.load()

            if width <= 0 or height <= 0:
                return ByteValidationResult(
                    is_valid=False,
                    decode_status="INVALID_DIMENSIONS",
                    width=width,
                    height=height,
                    image_format=fmt,
                    file_size_bytes=size,
                    sha256=sha256_hex,
                )

            pixel_hash = None
            if compute_pixel_hash:
                rgb = img.convert("RGB")
                pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()

            perceptual_hash = compute_dhash64(img) if compute_perceptual_hash else None

        return ByteValidationResult(
            is_valid=True,
            decode_status="OK",
            width=int(width),
            height=int(height),
            image_format=fmt,
            file_size_bytes=size,
            sha256=sha256_hex,
            pixel_hash=pixel_hash,
            perceptual_hash=perceptual_hash,
        )
    except Exception as exc:
        return ByteValidationResult(
            is_valid=False,
            decode_status=f"CORRUPTED:{type(exc).__name__}",
            width=None,
            height=None,
            image_format="CORRUPTED",
            file_size_bytes=size,
            sha256=sha256_hex,
        )
