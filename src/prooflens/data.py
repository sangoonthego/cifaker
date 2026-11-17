"""Dataset discovery, manifest generation, and quality auditing utilities for ProofLens.

This module implements Phase 0 dataset engineering:
- Recursive discovery of dataset images without hardcoded path assumptions
- Explicit mapping of generator folders, source splits, and REAL/FAKE binary labels
- Image header & pixel decoding verification and SHA-256 hashing
- Deterministic manifest generation (CSV)
- Comprehensive dataset quality, duplicate, and leakage auditing
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageFile
from tqdm import tqdm

# Ensure truncated images raise an error during load() instead of being silently accepted
ImageFile.LOAD_TRUNCATED_IMAGES = False

MANIFEST_COLUMNS = [
    "image_id",
    "relative_path",
    "source_split",
    "generator",
    "label",
    "label_name",
    "width",
    "height",
    "image_format",
    "file_extension",
    "file_size_bytes",
    "sha256",
    "decode_status",
]

DEFAULT_SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


@dataclass(frozen=True)
class ParsedPathMetadata:
    """Metadata extracted from an image's relative path within the dataset root."""

    relative_path: str
    generator_folder: str
    generator: str
    source_split: str
    raw_class_folder: str
    label: int
    label_name: str
    mapping_status: str


@dataclass
class ImageRecord:
    """Single row in the dataset manifest (`data/manifests/images.csv`)."""

    image_id: str
    relative_path: str
    source_split: str
    generator: str
    label: int
    label_name: str
    width: int | None
    height: int | None
    image_format: str
    file_extension: str
    file_size_bytes: int
    sha256: str
    decode_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml_config(config_path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def parse_relative_image_path(
    rel_path: Path | str,
    label_mapping: dict[str, dict[str, Any]] | None = None,
    split_mapping: dict[str, str] | None = None,
    generator_mapping: dict[str, str] | None = None,
) -> ParsedPathMetadata:
    """Parse relative image path into generator, split, and label without guessing.

    Expected standard Tiny GenImage hierarchy:
        <generator_dir>/<split_dir>/<class_dir>/<filename>
    Also supports optional wrapper directory:
        <wrapper_dir>/<generator_dir>/<split_dir>/<class_dir>/<filename>
    Or flat split hierarchy if generator is not nested:
        <split_dir>/<class_dir>/<filename>

    If any folder name is ambiguous or absent from the explicit mapping dictionary,
    this function never guesses: it records `UNKNOWN` / `-1` and sets `mapping_status`.
    """
    posix_rel = Path(rel_path).as_posix()
    parts = [p for p in posix_rel.split("/") if p and p != "."]

    if label_mapping is None:
        label_mapping = {
            "nature": {"label": 0, "label_name": "REAL"},
            "ai": {"label": 1, "label_name": "FAKE"},
        }
    if split_mapping is None:
        split_mapping = {"train": "train", "val": "val", "test": "test"}
    if generator_mapping is None:
        generator_mapping = {}

    if len(parts) < 3:
        return ParsedPathMetadata(
            relative_path=posix_rel,
            generator_folder="UNKNOWN",
            generator="UNKNOWN",
            source_split="UNKNOWN",
            raw_class_folder="UNKNOWN",
            label=-1,
            label_name="UNKNOWN",
            mapping_status="UNRESOLVED_SHALLOW_PATH",
        )

    # Identify split and class folders starting from the parent of the file
    raw_class_folder = parts[-2]
    raw_split_folder = parts[-3]
    raw_gen_folder = parts[-4] if len(parts) >= 4 else "UNKNOWN"

    # Normalize lookup keys (case-sensitive check first, then lowercase check if configured)
    class_info = label_mapping.get(raw_class_folder) or label_mapping.get(raw_class_folder.lower())
    split_name = split_mapping.get(raw_split_folder) or split_mapping.get(raw_split_folder.lower())

    issues: list[str] = []

    if class_info is not None:
        label = int(class_info["label"])
        label_name = str(class_info["label_name"])
    else:
        label = -1
        label_name = "UNKNOWN"
        issues.append(f"UNMAPPED_CLASS:{raw_class_folder}")

    if split_name is None:
        split_name = "UNKNOWN"
        issues.append(f"UNMAPPED_SPLIT:{raw_split_folder}")

    if raw_gen_folder == "UNKNOWN":
        generator_name = "UNKNOWN"
        issues.append("MISSING_GENERATOR_DIR")
    elif generator_mapping:
        if raw_gen_folder in generator_mapping:
            generator_name = str(generator_mapping[raw_gen_folder])
        else:
            generator_name = "UNKNOWN"
            issues.append(f"UNMAPPED_GENERATOR:{raw_gen_folder}")
    else:
        # When generator_mapping is empty/not enforced, preserve raw folder name if valid
        generator_name = raw_gen_folder

    status = "OK" if not issues else ";".join(issues)
    return ParsedPathMetadata(
        relative_path=posix_rel,
        generator_folder=raw_gen_folder,
        generator=generator_name,
        source_split=split_name,
        raw_class_folder=raw_class_folder,
        label=label,
        label_name=label_name,
        mapping_status=status,
    )


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_single_image(
    abs_path: Path,
    rel_path: str,
    label_mapping: dict[str, dict[str, Any]] | None = None,
    split_mapping: dict[str, str] | None = None,
    generator_mapping: dict[str, str] | None = None,
    supported_extensions: set[str] | None = None,
) -> tuple[ImageRecord, str, str]:
    """Inspect a single file on disk, computing hash, dimensions, format, and decode status.

    Returns:
        Tuple of (ImageRecord, image_mode, mapping_status).
    """
    if supported_extensions is None:
        supported_extensions = DEFAULT_SUPPORTED_EXTENSIONS

    parsed = parse_relative_image_path(
        rel_path=rel_path,
        label_mapping=label_mapping,
        split_mapping=split_mapping,
        generator_mapping=generator_mapping,
    )

    ext = abs_path.suffix.lower()
    try:
        file_size = abs_path.stat().st_size
    except OSError as exc:
        return (
            ImageRecord(
                image_id="",
                relative_path=parsed.relative_path,
                source_split=parsed.source_split,
                generator=parsed.generator,
                label=parsed.label,
                label_name=parsed.label_name,
                width=None,
                height=None,
                image_format="UNKNOWN",
                file_extension=ext,
                file_size_bytes=0,
                sha256="",
                decode_status=f"OS_ERROR:{type(exc).__name__}",
            ),
            "UNKNOWN",
            parsed.mapping_status,
        )

    if file_size == 0:
        sha256_hex = compute_sha256(abs_path)
        return (
            ImageRecord(
                image_id="",
                relative_path=parsed.relative_path,
                source_split=parsed.source_split,
                generator=parsed.generator,
                label=parsed.label,
                label_name=parsed.label_name,
                width=None,
                height=None,
                image_format="UNKNOWN",
                file_extension=ext,
                file_size_bytes=0,
                sha256=sha256_hex,
                decode_status="EMPTY_FILE",
            ),
            "UNKNOWN",
            parsed.mapping_status,
        )

    sha256_hex = compute_sha256(abs_path)

    if ext not in supported_extensions:
        return (
            ImageRecord(
                image_id="",
                relative_path=parsed.relative_path,
                source_split=parsed.source_split,
                generator=parsed.generator,
                label=parsed.label,
                label_name=parsed.label_name,
                width=None,
                height=None,
                image_format="UNSUPPORTED",
                file_extension=ext,
                file_size_bytes=file_size,
                sha256=sha256_hex,
                decode_status=f"UNSUPPORTED_EXTENSION:{ext}",
            ),
            "UNKNOWN",
            parsed.mapping_status,
        )

    width: int | None = None
    height: int | None = None
    img_format = "UNKNOWN"
    img_mode = "UNKNOWN"
    decode_status = "OK"

    try:
        with Image.open(abs_path) as img:
            img_format = (img.format or "UNKNOWN").upper()
            img_mode = img.mode or "UNKNOWN"
            width, height = img.size
            # Fully load pixel data to detect truncated or corrupt compressed streams
            img.load()
        if width is None or height is None or width <= 0 or height <= 0:
            decode_status = "INVALID_DIMENSIONS"
    except Exception as exc:
        decode_status = f"CORRUPTED:{type(exc).__name__}"

    if decode_status == "OK" and parsed.mapping_status != "OK":
        decode_status = f"OK_WITH_{parsed.mapping_status}"

    return (
        ImageRecord(
            image_id="",
            relative_path=parsed.relative_path,
            source_split=parsed.source_split,
            generator=parsed.generator,
            label=parsed.label,
            label_name=parsed.label_name,
            width=width,
            height=height,
            image_format=img_format,
            file_extension=ext,
            file_size_bytes=file_size,
            sha256=sha256_hex,
            decode_status=decode_status,
        ),
        img_mode,
        parsed.mapping_status,
    )


def discover_dataset_files(
    dataset_root: Path | str,
    ignore_names: set[str] | None = None,
) -> list[tuple[Path, str]]:
    """Recursively discover all files in `dataset_root` sorted deterministically by relative POSIX path."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if ignore_names is None:
        ignore_names = {".gitkeep", ".DS_Store", "Thumbs.db", "desktop.ini", ".acquisition_metadata.json"}

    discovered: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # Sort directories in-place for deterministic traversal
        dirnames.sort()
        for fname in sorted(filenames):
            if fname in ignore_names or fname.endswith(".zip") or fname.endswith(".part"):
                continue
            abs_p = Path(dirpath) / fname
            rel_p = abs_p.relative_to(root).as_posix()
            discovered.append((abs_p, rel_p))

    discovered.sort(key=lambda item: item[1])
    return discovered


def build_dataset_manifest(
    dataset_root: Path | str,
    config: dict[str, Any] | None = None,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scan `dataset_root`, validate every image, and return the manifest DataFrame and auxiliary audit data."""
    if config is None:
        config = {}

    label_mapping = config.get("label_mapping")
    split_mapping = config.get("split_mapping")
    generator_mapping = config.get("generator_mapping")
    supported_ext_list = config.get("supported_extensions")
    supported_extensions = (
        {e.lower() for e in supported_ext_list}
        if supported_ext_list
        else DEFAULT_SUPPORTED_EXTENSIONS
    )

    files = discover_dataset_files(dataset_root)
    records: list[ImageRecord] = []
    modes_counter: Counter[str] = Counter()
    modes_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved_mappings: list[dict[str, str]] = []

    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) * 2)

    def _worker(item: tuple[Path, str]) -> tuple[ImageRecord, str, str]:
        return inspect_single_image(
            abs_path=item[0],
            rel_path=item[1],
            label_mapping=label_mapping,
            split_mapping=split_mapping,
            generator_mapping=generator_mapping,
            supported_extensions=supported_extensions,
        )

    if max_workers <= 1 or len(files) < 50:
        iterator = (_worker(item) for item in files)
        if show_progress:
            iterator = tqdm(iterator, total=len(files), desc="Auditing images", unit="img")
        results = list(iterator)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            iterator = pool.map(_worker, files, chunksize=64)
            if show_progress:
                iterator = tqdm(iterator, total=len(files), desc="Auditing images", unit="img")
            results = list(iterator)

    # Sort deterministically by relative_path and assign stable sequential/hash-based image_id
    results.sort(key=lambda r: r[0].relative_path)

    rows: list[dict[str, Any]] = []
    for idx, (rec, mode, map_status) in enumerate(results, start=1):
        path_digest = hashlib.sha1(rec.relative_path.encode("utf-8")).hexdigest()[:10]
        image_id = f"img_{idx:06d}_{path_digest}"
        row = rec.to_dict()
        row["image_id"] = image_id
        rows.append(row)

        modes_counter[mode] += 1
        modes_by_label[rec.label_name][mode] += 1
        if map_status != "OK":
            unresolved_mappings.append(
                {"relative_path": rec.relative_path, "status": map_status}
            )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    aux_meta = {
        "image_modes": dict(modes_counter),
        "image_modes_by_label": {k: dict(v) for k, v in modes_by_label.items()},
        "unresolved_mappings": unresolved_mappings,
    }
    return df, aux_meta


def compute_audit_summary(
    df: pd.DataFrame,
    aux_meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    acquisition_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive Phase 0 dataset quality, integrity, duplicate, and leakage statistics."""
    if aux_meta is None:
        aux_meta = {}
    if config is None:
        config = {}
    if acquisition_info is None:
        acquisition_info = {}

    ds_cfg = config.get("dataset", {})
    total_files = len(df)

    if total_files == 0:
        return {
            "dataset_source": ds_cfg.get("kaggle_id", "yangsangtai/tiny-genimage"),
            "dataset_version": ds_cfg.get("expected_version", 1),
            "acquisition_status": acquisition_info.get("status", "EMPTY"),
            "file_count": 0,
            "valid_image_count": 0,
            "invalid_image_count": 0,
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    valid_mask = df["decode_status"] == "OK"
    valid_df = df[valid_mask]
    invalid_df = df[~valid_mask]

    # A. Dataset composition
    label_counts = {str(k): int(v) for k, v in df["label_name"].value_counts().to_dict().items()}
    generator_counts = {str(k): int(v) for k, v in df["generator"].value_counts().sort_index().to_dict().items()}
    split_counts = {str(k): int(v) for k, v in df["source_split"].value_counts().sort_index().to_dict().items()}

    class_by_generator: dict[str, dict[str, int]] = {}
    for gen, group in df.groupby("generator"):
        class_by_generator[str(gen)] = {
            str(k): int(v) for k, v in group["label_name"].value_counts().to_dict().items()
        }

    class_by_split: dict[str, dict[str, int]] = {}
    for sp, group in df.groupby("source_split"):
        class_by_split[str(sp)] = {
            str(k): int(v) for k, v in group["label_name"].value_counts().to_dict().items()
        }

    generator_split_class: dict[str, dict[str, dict[str, int]]] = {}
    for (gen, sp), group in df.groupby(["generator", "source_split"]):
        generator_split_class.setdefault(str(gen), {})[str(sp)] = {
            str(k): int(v) for k, v in group["label_name"].value_counts().to_dict().items()
        }

    # B. Image integrity breakdown
    decode_status_counts = {
        str(k): int(v) for k, v in df["decode_status"].value_counts().to_dict().items()
    }
    corrupted_files = invalid_df[invalid_df["decode_status"].str.startswith("CORRUPTED")]["relative_path"].tolist()
    empty_files = invalid_df[invalid_df["decode_status"] == "EMPTY_FILE"]["relative_path"].tolist()
    unsupported_files = invalid_df[invalid_df["decode_status"].str.startswith("UNSUPPORTED_EXTENSION")]["relative_path"].tolist()
    invalid_dim_files = invalid_df[invalid_df["decode_status"] == "INVALID_DIMENSIONS"]["relative_path"].tolist()

    # C. Image characteristics (sizes, aspect ratios, formats, file sizes)
    format_counts = {str(k): int(v) for k, v in df["image_format"].value_counts().to_dict().items()}
    extension_counts = {str(k): int(v) for k, v in df["file_extension"].value_counts().to_dict().items()}

    format_by_label: dict[str, dict[str, int]] = {}
    ext_by_label: dict[str, dict[str, int]] = {}
    for lbl, group in df.groupby("label_name"):
        format_by_label[str(lbl)] = {str(k): int(v) for k, v in group["image_format"].value_counts().to_dict().items()}
        ext_by_label[str(lbl)] = {str(k): int(v) for k, v in group["file_extension"].value_counts().to_dict().items()}

    format_by_generator_and_label: dict[str, dict[str, dict[str, int]]] = {}
    for (gen, lbl), group in df.groupby(["generator", "label_name"]):
        format_by_generator_and_label.setdefault(str(gen), {})[str(lbl)] = {
            str(k): int(v) for k, v in group["image_format"].value_counts().to_dict().items()
        }

    def _numeric_stats(series: pd.Series) -> dict[str, float]:
        clean = series.dropna().astype(float)
        if clean.empty:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "p05": 0.0, "p95": 0.0}
        return {
            "min": round(float(clean.min()), 2),
            "max": round(float(clean.max()), 2),
            "mean": round(float(clean.mean()), 2),
            "median": round(float(clean.median()), 2),
            "std": round(float(clean.std(ddof=0)), 2),
            "p05": round(float(clean.quantile(0.05)), 2),
            "p95": round(float(clean.quantile(0.95)), 2),
        }

    size_stats_overall = {
        "width": _numeric_stats(valid_df["width"]),
        "height": _numeric_stats(valid_df["height"]),
        "file_size_bytes": _numeric_stats(df["file_size_bytes"]),
    }
    if not valid_df.empty:
        aspect_ratios = valid_df["width"].astype(float) / valid_df["height"].astype(float)
        size_stats_overall["aspect_ratio"] = _numeric_stats(aspect_ratios)
        res_strings = valid_df["width"].astype(int).astype(str) + "x" + valid_df["height"].astype(int).astype(str)
        top_resolutions = {str(k): int(v) for k, v in res_strings.value_counts().head(15).to_dict().items()}
    else:
        size_stats_overall["aspect_ratio"] = _numeric_stats(pd.Series(dtype=float))
        top_resolutions = {}

    characteristics_by_label: dict[str, Any] = {}
    for lbl, group in valid_df.groupby("label_name"):
        ar = group["width"].astype(float) / group["height"].astype(float)
        res_s = group["width"].astype(int).astype(str) + "x" + group["height"].astype(int).astype(str)
        characteristics_by_label[str(lbl)] = {
            "count": int(len(group)),
            "width": _numeric_stats(group["width"]),
            "height": _numeric_stats(group["height"]),
            "aspect_ratio": _numeric_stats(ar),
            "file_size_bytes": _numeric_stats(group["file_size_bytes"]),
            "unique_resolutions": int(res_s.nunique()),
            "top_resolutions": {str(k): int(v) for k, v in res_s.value_counts().head(10).to_dict().items()},
        }

    characteristics_by_generator_label: dict[str, dict[str, Any]] = {}
    for (gen, lbl), group in valid_df.groupby(["generator", "label_name"]):
        res_s = group["width"].astype(int).astype(str) + "x" + group["height"].astype(int).astype(str)
        characteristics_by_generator_label.setdefault(str(gen), {})[str(lbl)] = {
            "count": int(len(group)),
            "width_mean": round(float(group["width"].mean()), 2),
            "height_mean": round(float(group["height"].mean()), 2),
            "file_size_kb_mean": round(float(group["file_size_bytes"].mean() / 1024.0), 2),
            "top_resolutions": {str(k): int(v) for k, v in res_s.value_counts().head(5).to_dict().items()},
        }

    # D. Duplicate analysis (Exact SHA-256)
    non_empty_sha = df[df["sha256"] != ""]
    sha_counts = non_empty_sha["sha256"].value_counts()
    dup_hashes = set(sha_counts[sha_counts > 1].index)
    dup_df = non_empty_sha[non_empty_sha["sha256"].isin(dup_hashes)]

    num_duplicate_groups = len(dup_hashes)
    total_files_in_dup_groups = int(len(dup_df))
    redundant_duplicate_copies = int(total_files_in_dup_groups - num_duplicate_groups)

    within_split_dup_groups = 0
    cross_split_dup_groups = 0
    cross_generator_dup_groups = 0
    cross_label_dup_groups = 0
    dup_groups_by_label: Counter[str] = Counter()
    dup_copies_by_label: Counter[str] = Counter()
    sample_duplicate_groups: list[dict[str, Any]] = []

    for sha, group in dup_df.groupby("sha256"):
        splits_in_grp = sorted(group["source_split"].unique().tolist())
        gens_in_grp = sorted(group["generator"].unique().tolist())
        labels_in_grp = sorted(group["label_name"].unique().tolist())

        if len(splits_in_grp) > 1:
            cross_split_dup_groups += 1
        else:
            within_split_dup_groups += 1

        if len(gens_in_grp) > 1:
            cross_generator_dup_groups += 1
        if len(labels_in_grp) > 1:
            cross_label_dup_groups += 1

        lbl_key = "+".join(labels_in_grp)
        dup_groups_by_label[lbl_key] += 1
        dup_copies_by_label[lbl_key] += int(len(group))

        if len(sample_duplicate_groups) < 25:
            sample_duplicate_groups.append(
                {
                    "sha256": str(sha),
                    "count": int(len(group)),
                    "splits": splits_in_grp,
                    "generators": gens_in_grp,
                    "labels": labels_in_grp,
                    "paths": sorted(group["relative_path"].tolist()),
                }
            )

    # Also check filename stem / ImageNet class prefix overlap across splits or generators
    duplicate_statistics = {
        "unique_sha256_count": int(non_empty_sha["sha256"].nunique()),
        "duplicate_sha256_groups": int(num_duplicate_groups),
        "total_files_in_duplicate_groups": total_files_in_dup_groups,
        "redundant_duplicate_files": redundant_duplicate_copies,
        "within_split_duplicate_groups": int(within_split_dup_groups),
        "cross_split_duplicate_groups": int(cross_split_dup_groups),
        "cross_generator_duplicate_groups": int(cross_generator_dup_groups),
        "cross_label_duplicate_groups": int(cross_label_dup_groups),
        "duplicate_groups_by_class": dict(dup_groups_by_label),
        "duplicate_files_by_class": dict(dup_copies_by_label),
        "near_duplicate_detection_implemented": False,
        "near_duplicate_limitation_note": (
            "Phase 0 implements exact cryptographic (SHA-256) duplicate detection only. "
            "Perceptual hashing (pHash/dHash) or embedding-based near-duplicate detection "
            "across resized/recompressed ImageNet crops is not performed in Phase 0."
        ),
        "sample_duplicate_groups": sample_duplicate_groups,
    }

    # E. Leakage & Confounding Risk Analysis
    integrity_warnings: list[str] = []
    if len(invalid_df) > 0:
        integrity_warnings.append(f"Detected {len(invalid_df)} invalid or unreadable files.")
    if cross_split_dup_groups > 0:
        integrity_warnings.append(
            f"CRITICAL LEAKAGE RISK: {cross_split_dup_groups} exact SHA-256 duplicate groups span across train and val splits."
        )
    if cross_generator_dup_groups > 0:
        integrity_warnings.append(
            f"CROSS-GENERATOR OVERLAP: {cross_generator_dup_groups} exact SHA-256 duplicate groups appear across multiple generator subsets."
        )
    if cross_label_dup_groups > 0:
        integrity_warnings.append(
            f"LABEL CONTRADICTION: {cross_label_dup_groups} exact SHA-256 duplicate groups have conflicting REAL and FAKE labels."
        )

    # Check format confounding (e.g., REAL = JPEG vs FAKE = PNG)
    real_formats = format_by_label.get("REAL", {})
    fake_formats = format_by_label.get("FAKE", {})
    format_confounding_detected = False
    if real_formats and fake_formats:
        real_top_fmt = max(real_formats, key=real_formats.get)
        fake_top_fmt = max(fake_formats, key=fake_formats.get)
        if real_top_fmt != fake_top_fmt:
            format_confounding_detected = True
            integrity_warnings.append(
                f"SEVERE FORMAT CONFOUNDING: REAL images are predominantly {real_top_fmt} ({real_formats}), "
                f"whereas FAKE images are predominantly {fake_top_fmt} ({fake_formats}). "
                "A CNN may learn JPEG compression artifacts vs lossless PNG encoding shortcuts rather than generative forensic traces."
            )

    # Check resolution confounding
    resolution_confounding_notes: list[str] = []
    for gen, gen_dict in characteristics_by_generator_label.items():
        fake_top = list(gen_dict.get("FAKE", {}).get("top_resolutions", {}).items())
        real_top = list(gen_dict.get("REAL", {}).get("top_resolutions", {}).items())
        if fake_top:
            top_res_str, top_res_cnt = fake_top[0]
            fake_cnt = gen_dict["FAKE"]["count"]
            if top_res_cnt / max(fake_cnt, 1) > 0.8:
                resolution_confounding_notes.append(
                    f"Generator '{gen}' FAKE images are {top_res_cnt}/{fake_cnt} ({top_res_cnt/fake_cnt:.1%}) fixed at {top_res_str}, "
                    f"while paired REAL images have diverse resolutions (top: {real_top[:2]})."
                )
    if resolution_confounding_notes:
        integrity_warnings.append(
            "RESOLUTION BIAS: FAKE images exhibit generator-specific fixed resolutions while REAL ImageNet images have variable dimensions."
        )

    missing_metadata = {
        "missing_width_count": int(df["width"].isna().sum()),
        "missing_height_count": int(df["height"].isna().sum()),
        "unknown_label_count": int((df["label"] == -1).sum()),
        "unknown_generator_count": int((df["generator"] == "UNKNOWN").sum()),
        "unknown_split_count": int((df["source_split"] == "UNKNOWN").sum()),
        "unresolved_path_mappings": len(aux_meta.get("unresolved_mappings", [])),
    }

    summary = {
        "dataset_source": ds_cfg.get("kaggle_id", "yangsangtai/tiny-genimage"),
        "dataset_url": "https://www.kaggle.com/datasets/yangsangtai/tiny-genimage",
        "dataset_version": acquisition_info.get("dataset_version", ds_cfg.get("expected_version", 1)),
        "license": acquisition_info.get("license", ds_cfg.get("license", "CC BY-NC-SA 4.0")),
        "acquisition_status": acquisition_info.get("status", "VERIFIED"),
        "acquisition_details": acquisition_info,
        "file_count": int(total_files),
        "valid_image_count": int(len(valid_df)),
        "invalid_image_count": int(len(invalid_df)),
        "label_counts": label_counts,
        "generator_counts": generator_counts,
        "split_counts": split_counts,
        "class_distribution_by_generator": class_by_generator,
        "class_distribution_by_split": class_by_split,
        "generator_split_class_breakdown": generator_split_class,
        "decode_status_counts": decode_status_counts,
        "integrity_failures": {
            "corrupted_files": corrupted_files,
            "empty_files": empty_files,
            "unsupported_files": unsupported_files,
            "invalid_dimension_files": invalid_dim_files,
        },
        "image_sizes": {
            "overall": size_stats_overall,
            "top_resolutions_overall": top_resolutions,
            "by_label": characteristics_by_label,
            "by_generator_and_label": characteristics_by_generator_label,
        },
        "image_formats": {
            "overall_formats": format_counts,
            "overall_extensions": extension_counts,
            "formats_by_label": format_by_label,
            "extensions_by_label": ext_by_label,
            "formats_by_generator_and_label": format_by_generator_and_label,
            "image_modes_overall": aux_meta.get("image_modes", {}),
            "image_modes_by_label": aux_meta.get("image_modes_by_label", {}),
        },
        "duplicate_statistics": duplicate_statistics,
        "leakage_risk_analysis": {
            "exact_cross_split_duplicate_groups": int(cross_split_dup_groups),
            "exact_cross_generator_duplicate_groups": int(cross_generator_dup_groups),
            "exact_cross_label_duplicate_groups": int(cross_label_dup_groups),
            "format_confounding_detected": format_confounding_detected,
            "resolution_confounding_notes": resolution_confounding_notes,
        },
        "integrity_warnings": integrity_warnings,
        "missing_metadata": missing_metadata,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return summary
