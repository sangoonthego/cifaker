"""Safe, reproducible Kaggle dataset acquisition utility for ProofLens (Phase 0).

Downloads and extracts the complete `yangsangtai/tiny-genimage` dataset from Kaggle
into `data/raw/tiny_genimage/`, performing pre-flight environment, credential,
existing-file, and disk-space checks.

Supports:
1. Official Kaggle CLI (`kaggle datasets download`) when authenticated.
2. Direct Kaggle API v1 download + on-the-fly streaming ZIP64 extraction when
   disk space cannot hold both the 7.77 GB `.zip` archive and the 7.78 GB extracted
   files simultaneously, or when `kaggle.json` is not configured for a public dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure local package is importable when executed directly as `python scripts/download_dataset.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prooflens.data import discover_dataset_files, load_yaml_config


def check_kaggle_auth_and_cli() -> dict[str, Any]:
    """Check whether Kaggle CLI and authentication credentials exist without exposing secrets."""
    cli_path = shutil.which("kaggle")
    if cli_path is None:
        # Check inside current Python environment Scripts directory
        venv_kaggle = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
        if venv_kaggle.is_file():
            cli_path = str(venv_kaggle)

    home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
    config_dir_env = os.environ.get("KAGGLE_CONFIG_DIR")
    custom_kaggle = Path(config_dir_env) / "kaggle.json" if config_dir_env else None
    has_env_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))

    has_file_creds = home_kaggle.is_file() or (custom_kaggle is not None and custom_kaggle.is_file())
    authenticated = has_file_creds or has_env_creds or has_token

    return {
        "kaggle_cli_available": cli_path is not None,
        "kaggle_cli_executable": cli_path is not None,
        "authenticated": authenticated,
        "auth_method": (
            "env_vars"
            if (has_env_creds or has_token)
            else ("kaggle_json" if has_file_creds else "unauthenticated_public_api")
        ),
    }


def fetch_kaggle_dataset_metadata(dataset_id: str) -> dict[str, Any]:
    """Query Kaggle public API v1 for dataset version, license, and size metadata."""
    url = f"https://www.kaggle.com/api/v1/datasets/view/{dataset_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {
        "dataset_id": data.get("ref", dataset_id),
        "title": data.get("title") or data.get("titleNullable"),
        "creator": data.get("creatorName") or data.get("creatorNameNullable"),
        "version": data.get("currentVersionNumber") or data.get("currentVersionNumberNullable"),
        "license": data.get("licenseName") or data.get("licenseNameNullable"),
        "last_updated": data.get("lastUpdated"),
        "total_uncompressed_bytes": data.get("totalBytes") or data.get("totalBytesNullable"),
        "is_private": data.get("isPrivate", False),
    }


def fetch_remote_zip64_inventory(download_url: str) -> dict[str, Any]:
    """Inspect the remote ZIP64 Central Directory via HTTP Range requests to get the exact file inventory."""
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        signed_url = resp.url
        archive_bytes = int(resp.headers.get("Content-Length", 0))
        etag = resp.headers.get("ETag", "").strip('"')
        goog_hash = resp.headers.get("x-goog-hash", "")

    # Fetch last 64 KB to locate ZIP64 End of Central Directory
    tail_size = min(65536, archive_bytes)
    req_tail = urllib.request.Request(
        signed_url,
        headers={"Range": f"bytes={archive_bytes - tail_size}-{archive_bytes - 1}"},
    )
    with urllib.request.urlopen(req_tail, timeout=20) as resp:
        tail = resp.read()

    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos == -1:
        raise RuntimeError("Could not locate ZIP End of Central Directory in remote archive.")

    eocd64_loc = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
    if eocd64_loc != -1:
        _, _, eocd64_offset, _ = struct.unpack("<4sIQI", tail[eocd64_loc : eocd64_loc + 20])
        req_eocd64 = urllib.request.Request(
            signed_url,
            headers={"Range": f"bytes={eocd64_offset}-{archive_bytes - 1}"},
        )
        with urllib.request.urlopen(req_eocd64, timeout=20) as resp:
            eocd64_data = resp.read()
        (
            _sig,
            _sz,
            _ver_made,
            _ver_need,
            _disk,
            _cd_disk,
            _entries_disk,
            entries_total,
            cd_size,
            cd_offset,
        ) = struct.unpack("<4sQHHIIQQQQ", eocd64_data[:56])
    else:
        _, _, _, entries_total, cd_size, cd_offset, _ = struct.unpack(
            "<4sHHHHIIH", tail[eocd_pos : eocd_pos + 22]
        )

    req_cd = urllib.request.Request(
        signed_url,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
    )
    with urllib.request.urlopen(req_cd, timeout=60) as resp:
        cd_data = resp.read()

    pos = 0
    entries: dict[str, dict[str, int]] = {}
    uncompressed_sum = 0
    while pos < len(cd_data):
        if cd_data[pos : pos + 4] != b"PK\x01\x02":
            break
        method, _mtime, _mdate, crc32_val, comp_sz, uncomp_sz, fn_len, extra_len, comm_len = (
            struct.unpack("<HIIIIIHHH", cd_data[pos + 10 : pos + 34])
        )
        fn = cd_data[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="replace")
        extra = cd_data[pos + 46 + fn_len : pos + 46 + fn_len + extra_len]
        epos = 0
        while epos + 4 <= len(extra):
            hid, dlen = struct.unpack("<HH", extra[epos : epos + 4])
            if hid == 1:
                vals = struct.unpack("<" + "Q" * (dlen // 8), extra[epos + 4 : epos + 4 + dlen])
                idx = 0
                if uncomp_sz == 0xFFFFFFFF and idx < len(vals):
                    uncomp_sz = vals[idx]
                    idx += 1
                if comp_sz == 0xFFFFFFFF and idx < len(vals):
                    comp_sz = vals[idx]
                    idx += 1
                break
            epos += 4 + dlen

        if not fn.endswith("/"):
            entries[fn] = {
                "crc32": crc32_val,
                "compressed_size": comp_sz,
                "uncompressed_size": uncomp_sz,
                "method": method,
            }
            uncompressed_sum += uncomp_sz
        pos += 46 + fn_len + extra_len + comm_len

    return {
        "archive_size_bytes": archive_bytes,
        "etag_md5": etag,
        "goog_hash": goog_hash,
        "expected_file_count": len(entries),
        "expected_uncompressed_bytes": uncompressed_sum,
        "entries": entries,
    }


def stream_download_and_extract_zip(
    download_url: str,
    target_dir: Path,
    cd_entries: dict[str, dict[str, int]],
    expected_archive_bytes: int,
) -> dict[str, Any]:
    """Stream the Kaggle dataset ZIP archive over HTTP, computing archive SHA-256 and extracting all files on the fly.

    Uses the Central Directory metadata (`cd_entries`) to know exact compressed/uncompressed sizes
    and verify CRC-32 checksums of every extracted file without needing 2x disk space.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_sha256 = hashlib.sha256()
    bytes_downloaded = 0
    extracted_count = 0
    skipped_existing_count = 0
    crc_mismatches: list[str] = []

    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        buf = bytearray()

        def read_exact(n: int) -> bytes:
            nonlocal bytes_downloaded
            while len(buf) < n:
                chunk = resp.read(max(1024 * 1024, n - len(buf)))
                if not chunk:
                    raise EOFError(
                        f"Unexpected EOF while streaming ZIP archive at byte {bytes_downloaded} "
                        f"(needed {n} bytes, buffer has {len(buf)})."
                    )
                archive_sha256.update(chunk)
                bytes_downloaded += len(chunk)
                buf.extend(chunk)
            out = bytes(buf[:n])
            del buf[:n]
            return out

        total_expected = len(cd_entries)
        print(
            f"[ProofLens] Streaming & extracting {total_expected:,} files "
            f"({expected_archive_bytes / (1024**3):.2f} GB compressed) into {target_dir}...",
            flush=True,
        )

        while True:
            sig = read_exact(4)
            if sig == b"PK\x01\x02" or sig == b"PK\x06\x06" or sig == b"PK\x05\x06":
                # Reached Central Directory at end of ZIP stream; consume remaining bytes for full archive SHA-256
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    archive_sha256.update(chunk)
                    bytes_downloaded += len(chunk)
                break

            if sig != b"PK\x03\x04":
                raise RuntimeError(f"Unexpected ZIP header signature: {sig!r} at byte {bytes_downloaded}")

            header_bytes = read_exact(26)
            (
                _ver,
                flags,
                method,
                _mtime,
                _mdate,
                local_crc,
                local_comp_sz,
                local_uncomp_sz,
                fn_len,
                extra_len,
            ) = struct.unpack("<HHHHHIIIHH", header_bytes)

            fn_bytes = read_exact(fn_len)
            extra_bytes = read_exact(extra_len)
            rel_name = fn_bytes.decode("utf-8", errors="replace")

            # Resolve exact sizes from Central Directory if ZIP64 (0xFFFFFFFF) or data descriptor flag is set
            cd_info = cd_entries.get(rel_name)
            if cd_info is not None:
                comp_sz = cd_info["compressed_size"]
                uncomp_sz = cd_info["uncompressed_size"]
                expected_crc = cd_info["crc32"]
                method = cd_info["method"]
            else:
                comp_sz = local_comp_sz
                uncomp_sz = local_uncomp_sz
                expected_crc = local_crc
                if comp_sz == 0xFFFFFFFF or uncomp_sz == 0xFFFFFFFF:
                    epos = 0
                    while epos + 4 <= len(extra_bytes):
                        hid, dlen = struct.unpack("<HH", extra_bytes[epos : epos + 4])
                        if hid == 1:
                            vals = struct.unpack("<" + "Q" * (dlen // 8), extra_bytes[epos + 4 : epos + 4 + dlen])
                            if uncomp_sz == 0xFFFFFFFF and len(vals) >= 1:
                                uncomp_sz = vals[0]
                            if comp_sz == 0xFFFFFFFF and len(vals) >= 2:
                                comp_sz = vals[1]
                            break
                        epos += 4 + dlen

            if rel_name.endswith("/"):
                if comp_sz > 0:
                    _ = read_exact(comp_sz)
                (target_dir / rel_name).mkdir(parents=True, exist_ok=True)
                continue

            out_path = (target_dir / rel_name).resolve()
            # Prevent path traversal outside target_dir
            if not str(out_path).startswith(str(target_dir.resolve())):
                raise RuntimeError(f"Unsafe path traversal detected in archive member: {rel_name}")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if valid file already exists on disk
            if out_path.is_file() and out_path.stat().st_size == uncomp_sz:
                # Still need to advance the stream past compressed payload
                remaining = comp_sz
                while remaining > 0:
                    step = min(1024 * 1024, remaining)
                    _ = read_exact(step)
                    remaining -= step
                skipped_existing_count += 1
            else:
                if method == 0:
                    # Stored (uncompressed)
                    crc_calc = 0
                    remaining = comp_sz
                    with out_path.open("wb") as out_f:
                        while remaining > 0:
                            step = min(1024 * 1024, remaining)
                            chunk = read_exact(step)
                            crc_calc = zlib.crc32(chunk, crc_calc)
                            out_f.write(chunk)
                            remaining -= step
                elif method == 8:
                    # Deflated
                    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                    crc_calc = 0
                    remaining = comp_sz
                    with out_path.open("wb") as out_f:
                        while remaining > 0:
                            step = min(1024 * 1024, remaining)
                            chunk = read_exact(step)
                            decompressed = decompressor.decompress(chunk)
                            if decompressed:
                                crc_calc = zlib.crc32(decompressed, crc_calc)
                                out_f.write(decompressed)
                            remaining -= step
                        tail_dec = decompressor.flush()
                        if tail_dec:
                            crc_calc = zlib.crc32(tail_dec, crc_calc)
                            out_f.write(tail_dec)
                else:
                    raise RuntimeError(f"Unsupported ZIP compression method {method} for {rel_name}")

                if (crc_calc & 0xFFFFFFFF) != (expected_crc & 0xFFFFFFFF):
                    crc_mismatches.append(rel_name)
                extracted_count += 1

            # Handle optional Data Descriptor (Bit 3 of general purpose flags)
            if flags & 0x08:
                dd_peek = read_exact(4)
                if dd_peek == b"PK\x07\x08":
                    # 16-byte or 24-byte (ZIP64) descriptor; check if ZIP64 extra was present
                    _ = read_exact(20 if b"\x01\x00" in extra_bytes[:4] else 12)
                else:
                    # No signature; 12 or 20 bytes
                    _ = read_exact(16 if b"\x01\x00" in extra_bytes[:4] else 8)

            total_done = extracted_count + skipped_existing_count
            if total_done % 2500 == 0 or total_done == total_expected:
                pct = (bytes_downloaded / max(expected_archive_bytes, 1)) * 100
                print(
                    f"[ProofLens] Progress: {total_done:,}/{total_expected:,} files "
                    f"({bytes_downloaded / (1024**3):.2f} GB / {expected_archive_bytes / (1024**3):.2f} GB, {pct:.1f}%)",
                    flush=True,
                )

    return {
        "archive_bytes_downloaded": bytes_downloaded,
        "archive_sha256": archive_sha256.hexdigest(),
        "extracted_count": extracted_count,
        "skipped_existing_count": skipped_existing_count,
        "crc_mismatches": crc_mismatches,
    }


def verify_extracted_inventory(target_dir: Path, cd_entries: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Verify that every expected file from the Kaggle archive inventory exists on disk with matching byte size."""
    discovered = discover_dataset_files(target_dir)
    discovered_map = {rel_p: abs_p.stat().st_size for abs_p, rel_p in discovered}

    missing_files: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    for rel_p, info in cd_entries.items():
        if rel_p not in discovered_map:
            missing_files.append(rel_p)
        elif discovered_map[rel_p] != info["uncompressed_size"]:
            size_mismatches.append(
                {
                    "relative_path": rel_p,
                    "expected_bytes": info["uncompressed_size"],
                    "actual_bytes": discovered_map[rel_p],
                }
            )

    extra_files = sorted(set(discovered_map.keys()) - set(cd_entries.keys()))
    total_disk_bytes = sum(discovered_map.values())

    return {
        "verified_complete": len(missing_files) == 0 and len(size_mismatches) == 0,
        "expected_file_count": len(cd_entries),
        "discovered_file_count": len(discovered_map),
        "total_extracted_bytes": total_disk_bytes,
        "missing_file_count": len(missing_files),
        "missing_files_sample": missing_files[:20],
        "size_mismatch_count": len(size_mismatches),
        "size_mismatches_sample": size_mismatches[:20],
        "extra_file_count": len(extra_files),
        "extra_files_sample": extra_files[:20],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the complete Tiny GenImage dataset from Kaggle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to dataset YAML configuration file (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override Kaggle dataset identifier (default from config: yangsangtai/tiny-genimage).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override target extraction directory (default from config: data/raw/tiny_genimage).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if existing dataset passes completeness verification.",
    )
    parser.add_argument(
        "--prefer-cli",
        action="store_true",
        help="Prefer invoking the `kaggle` CLI subprocess when credentials and 2x disk space are available.",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml_config(args.config) if args.config.is_file() else {}
    ds_cfg = cfg.get("dataset", {})
    dataset_id = args.dataset or ds_cfg.get("kaggle_id", "yangsangtai/tiny-genimage")
    target_dir = Path(args.output_dir or ds_cfg.get("raw_dir", "data/raw/tiny_genimage"))
    target_dir.mkdir(parents=True, exist_ok=True)

    auth_status = check_kaggle_auth_and_cli()
    print(f"[ProofLens] Dataset Identifier: {dataset_id}")
    print(f"[ProofLens] Target Directory:   {target_dir} (Resolved: {target_dir.resolve()})")
    print(
        f"[ProofLens] Kaggle CLI Ready:   {auth_status['kaggle_cli_available']} | "
        f"Auth Mode: {auth_status['auth_method']}"
    )

    # Query Kaggle metadata & remote inventory
    download_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
    try:
        api_meta = fetch_kaggle_dataset_metadata(dataset_id)
        remote_inv = fetch_remote_zip64_inventory(download_url)
    except Exception as exc:
        print(f"[ProofLens] ERROR querying Kaggle dataset metadata: {exc}", file=sys.stderr)
        return 1

    print(
        f"[ProofLens] Verified Kaggle Metadata: version={api_meta['version']}, "
        f"license='{api_meta['license']}', expected_files={remote_inv['expected_file_count']:,}, "
        f"archive_size={remote_inv['archive_size_bytes'] / (1024**3):.3f} GB, "
        f"uncompressed_size={remote_inv['expected_uncompressed_bytes'] / (1024**3):.3f} GB"
    )

    # Check if dataset is already present and complete
    pre_check = verify_extracted_inventory(target_dir, remote_inv["entries"])
    meta_record_path = target_dir / ".acquisition_metadata.json"

    if pre_check["verified_complete"] and not args.force:
        print(
            f"[ProofLens] Existing dataset in {target_dir} is already COMPLETE "
            f"({pre_check['discovered_file_count']:,}/{pre_check['expected_file_count']:,} files, "
            f"{pre_check['total_extracted_bytes'] / (1024**3):.3f} GB). Skipping duplicate download."
        )
        if not meta_record_path.is_file():
            record = {
                "status": "VERIFIED_COMPLETE",
                "dataset_id": dataset_id,
                "dataset_version": api_meta["version"],
                "license": api_meta["license"],
                "last_updated": api_meta["last_updated"],
                "acquisition_date_utc": datetime.now(timezone.utc).isoformat(),
                "download_method": "existing_verified",
                "archive_filename": "tiny-genimage.zip",
                "archive_size_bytes": remote_inv["archive_size_bytes"],
                "archive_etag_md5": remote_inv["etag_md5"],
                "archive_goog_hash": remote_inv["goog_hash"],
                "extracted_size_bytes": pre_check["total_extracted_bytes"],
                "discovered_image_files": pre_check["discovered_file_count"],
                "expected_image_files": pre_check["expected_file_count"],
                "missing_files": 0,
                "failed_files": 0,
            }
            meta_record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return 0

    # Check available disk space on the resolved target volume
    disk_usage = shutil.disk_usage(target_dir.resolve())
    needed_bytes = remote_inv["expected_uncompressed_bytes"] - pre_check["total_extracted_bytes"]
    print(
        f"[ProofLens] Disk Space on Target Volume: {disk_usage.free / (1024**3):.2f} GB free "
        f"(additional space needed for extraction: {max(0, needed_bytes) / (1024**3):.2f} GB)"
    )

    if disk_usage.free < needed_bytes + 500 * 1024 * 1024:
        print(
            f"[ProofLens] BLOCKED: Insufficient free disk space on {target_dir.resolve().anchor} "
            f"({disk_usage.free / (1024**3):.2f} GB free, require at least "
            f"{(needed_bytes + 500 * 1024 * 1024) / (1024**3):.2f} GB).",
            file=sys.stderr,
        )
        return 2

    # Choose between Kaggle CLI subprocess (requires auth + 2x space for zip + unzip) or streaming Kaggle API v1
    can_use_cli = (
        args.prefer_cli
        and auth_status["kaggle_cli_available"]
        and auth_status["authenticated"]
        and disk_usage.free > (remote_inv["archive_size_bytes"] + needed_bytes + 1024**3)
    )

    download_method = ""
    stream_stats: dict[str, Any] = {}

    if can_use_cli:
        download_method = "kaggle_cli_download_unzip"
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_id,
            "-p",
            str(target_dir),
            "--unzip",
        ]
        print(f"[ProofLens] Executing Kaggle CLI: {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print("[ProofLens] Kaggle CLI failed; falling back to direct Kaggle API v1 streaming extractor.")
            can_use_cli = False

    if not can_use_cli:
        download_method = "kaggle_api_v1_stream_unzip"
        stream_stats = stream_download_and_extract_zip(
            download_url=download_url,
            target_dir=target_dir,
            cd_entries=remote_inv["entries"],
            expected_archive_bytes=remote_inv["archive_size_bytes"],
        )

    post_check = verify_extracted_inventory(target_dir, remote_inv["entries"])
    status_str = (
        "VERIFIED_COMPLETE"
        if post_check["verified_complete"] and not stream_stats.get("crc_mismatches")
        else "INCOMPLETE_OR_MISMATCH"
    )

    record = {
        "status": status_str,
        "dataset_id": dataset_id,
        "dataset_version": api_meta["version"],
        "license": api_meta["license"],
        "creator": api_meta["creator"],
        "last_updated": api_meta["last_updated"],
        "acquisition_date_utc": datetime.now(timezone.utc).isoformat(),
        "download_method": download_method,
        "archive_filename": "tiny-genimage.zip",
        "archive_size_bytes": remote_inv["archive_size_bytes"],
        "archive_etag_md5": remote_inv["etag_md5"],
        "archive_goog_hash": remote_inv["goog_hash"],
        "archive_sha256": stream_stats.get("archive_sha256", ""),
        "extracted_size_bytes": post_check["total_extracted_bytes"],
        "discovered_image_files": post_check["discovered_file_count"],
        "expected_image_files": post_check["expected_file_count"],
        "missing_files": post_check["missing_file_count"],
        "size_mismatches": post_check["size_mismatch_count"],
        "crc_mismatches": len(stream_stats.get("crc_mismatches", [])),
    }
    meta_record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"[ProofLens] Acquisition Status: {status_str}")
    print(
        f"[ProofLens] Verified Files:     {post_check['discovered_file_count']:,} / "
        f"{post_check['expected_file_count']:,} ({post_check['total_extracted_bytes'] / (1024**3):.3f} GB)"
    )
    if stream_stats.get("archive_sha256"):
        print(f"[ProofLens] Archive SHA-256:    {stream_stats['archive_sha256']}")

    return 0 if status_str == "VERIFIED_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
