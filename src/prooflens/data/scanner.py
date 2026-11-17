"""Dataset directory scanner for ProofLens.

Recursively scans a dataset directory tree, parses folder-to-label, split, and
generator metadata using explicit mappings (rejecting ambiguous mappings), and
produces a validated manifest DataFrame.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prooflens.data.manifest import (
    ALL_MANIFEST_COLUMNS,
    LABEL_FAKE,
    LABEL_REAL,
    ManifestRow,
    validate_manifest_dataframe,
)
from prooflens.data.validation import DEFAULT_SUPPORTED_EXTENSIONS, verify_image_file

DEFAULT_LABEL_MAPPING: dict[str, int] = {
    "nature": LABEL_REAL,
    "real": LABEL_REAL,
    "0": LABEL_REAL,
    "ai": LABEL_FAKE,
    "fake": LABEL_FAKE,
    "1": LABEL_FAKE,
}

DEFAULT_SPLIT_MAPPING: dict[str, str] = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


@dataclass(frozen=True)
class ParsedRelativePath:
    """Structured metadata parsed from a relative image path."""

    relative_path: str
    generator: str
    source_split: str
    label: int


def parse_relative_image_path(
    relative_path: Path | str,
    label_mapping: dict[str, int] | None = None,
    split_mapping: dict[str, str] | None = None,
    generator_mapping: dict[str, str] | None = None,
    strict_generator_mapping: bool = False,
) -> ParsedRelativePath:
    """Parse a relative path into `(generator, source_split, label)` without guessing.

    Supported directory layouts:
    1. `<generator>/<split>/<class>/<filename>` (e.g. `biggan/train/ai/img.png`)
    2. `<split>/<generator>/<class>/<filename>` (e.g. `train/biggan/ai/img.png`)
    3. `<split>/<class>/<filename>` (when generator is not in path -> `generator="unknown"`)

    Raises:
        ValueError: If the class folder or split folder is missing or ambiguous (not in explicit mappings),
            or if `strict_generator_mapping=True` and the generator folder is unmapped.
    """
    posix_path = Path(relative_path).as_posix()
    parts = [p for p in posix_path.split("/") if p and p != "."]

    if len(parts) < 3:
        raise ValueError(
            f"Ambiguous relative path '{posix_path}': expected at least <split>/<class>/<file>."
        )

    lbl_map = (
        {k.lower(): int(v) for k, v in label_mapping.items()}
        if label_mapping is not None
        else DEFAULT_LABEL_MAPPING
    )
    sp_map = (
        {k.lower(): str(v) for k, v in split_mapping.items()}
        if split_mapping is not None
        else DEFAULT_SPLIT_MAPPING
    )
    gen_map = generator_mapping or {}

    raw_class = parts[-2].lower()
    if raw_class not in lbl_map:
        raise ValueError(
            f"Ambiguous or unmapped class folder '{parts[-2]}' in path '{posix_path}'. "
            f"Explicit label_mapping keys: {sorted(lbl_map.keys())}"
        )
    label = lbl_map[raw_class]

    if len(parts) == 3:
        raw_split = parts[-3].lower()
        if raw_split not in sp_map:
            raise ValueError(
                f"Ambiguous or unmapped split folder '{parts[-3]}' in path '{posix_path}'."
            )
        source_split = sp_map[raw_split]
        raw_generator = "unknown"
    else:
        cand_a = parts[-3]
        cand_b = parts[-4]
        if cand_a.lower() in sp_map:
            source_split = sp_map[cand_a.lower()]
            raw_generator = cand_b
        elif cand_b.lower() in sp_map:
            source_split = sp_map[cand_b.lower()]
            raw_generator = cand_a
        else:
            raise ValueError(
                f"Ambiguous split hierarchy in '{posix_path}': neither '{cand_a}' nor '{cand_b}' "
                f"is in explicit split_mapping {sorted(sp_map.keys())}."
            )

    if raw_generator in gen_map:
        generator = str(gen_map[raw_generator])
    elif raw_generator.lower() in gen_map:
        generator = str(gen_map[raw_generator.lower()])
    elif strict_generator_mapping:
        raise ValueError(
            f"Unmapped generator folder '{raw_generator}' in '{posix_path}' under strict_generator_mapping."
        )
    else:
        generator = raw_generator if raw_generator else "unknown"

    return ParsedRelativePath(
        relative_path=posix_path,
        generator=generator,
        source_split=source_split,
        label=label,
    )


def scan_dataset_directory(
    dataset_root: Path | str,
    label_mapping: dict[str, int] | None = None,
    split_mapping: dict[str, str] | None = None,
    generator_mapping: dict[str, str] | None = None,
    strict_generator_mapping: bool = False,
    supported_extensions: set[str] | frozenset[str] = DEFAULT_SUPPORTED_EXTENSIONS,
    include_corrupted: bool = False,
) -> pd.DataFrame:
    """Scan `dataset_root` deterministically and construct a manifest DataFrame.

    Args:
        dataset_root: Root directory to scan.
        label_mapping: Explicit folder-to-binary-label mapping (`0` for REAL, `1` for FAKE).
        split_mapping: Explicit folder-to-split mapping (`train`, `val`, `test`).
        generator_mapping: Optional mapping from raw directory names to canonical generator IDs.
        strict_generator_mapping: If True, reject directories not present in `generator_mapping`.
        supported_extensions: Allowed lowercase file extensions.
        include_corrupted: If True, include corrupted/unreadable files in the returned DataFrame
            with `decode_status != "OK"`; if False, skip or raise as configured.

    Returns:
        Validated manifest DataFrame conforming to `ALL_MANIFEST_COLUMNS`.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    ignore_files = {
        ".gitkeep",
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        ".acquisition_metadata.json",
    }
    discovered: list[tuple[Path, str]] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        for fname in sorted(filenames):
            if fname in ignore_files or fname.startswith("."):
                continue
            abs_p = Path(dirpath) / fname
            if abs_p.suffix.lower() not in supported_extensions and not include_corrupted:
                continue
            rel_p = abs_p.relative_to(root).as_posix()
            discovered.append((abs_p, rel_p))

    discovered.sort(key=lambda item: item[1])

    rows: list[dict[str, Any]] = []
    for abs_p, rel_p in discovered:
        parsed = parse_relative_image_path(
            relative_path=rel_p,
            label_mapping=label_mapping,
            split_mapping=split_mapping,
            generator_mapping=generator_mapping,
            strict_generator_mapping=strict_generator_mapping,
        )
        integrity = verify_image_file(abs_p, supported_extensions=supported_extensions)
        if not integrity.is_valid and not include_corrupted:
            continue

        path_hash = hashlib.sha1(parsed.relative_path.encode("utf-8")).hexdigest()[:12]
        image_id = f"img_{path_hash}"

        row = ManifestRow(
            image_id=image_id,
            relative_path=parsed.relative_path,
            generator=parsed.generator,
            label=parsed.label,
            source_split=parsed.source_split,
            width=integrity.width,
            height=integrity.height,
            image_format=integrity.image_format,
            file_size_bytes=integrity.file_size_bytes,
            sha256=integrity.sha256,
            decode_status=integrity.decode_status,
        )
        rows.append(row.to_dict())

    df = pd.DataFrame(rows, columns=list(ALL_MANIFEST_COLUMNS))
    return validate_manifest_dataframe(df)
