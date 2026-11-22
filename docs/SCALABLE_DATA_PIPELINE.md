# ProofLens — Scalable GenImage Data Pipeline Guide

This guide documents the production-oriented dataset engineering pipeline built inside `src/prooflens/data/` to construct, verify, publish, scale, and consume versioned GenImage dataset releases (120K → 200K → 500K+ images) across local development, GitHub, Google Colab, and Google Drive (`cifaker`).

---

## 1. Architecture Overview

```text
LOCAL DEV & CI (Windows/Linux CPU, synthetic fixtures only)
  -> Version-controlled Python package (src/prooflens/data/)
  -> GitHub Repository
  -> Google Colab Execution Runtime
     1. Source Inventory     : SourceAdapter -> Bounded-memory Parquet + summary JSON
     2. Validation & Leakage : SHA-256 / pixel_hash / Band-LSH dHash -> LeakageGuard
     3. Deterministic Plan   : Nested sampling (Train(120K) ⊆ Train(200K) ⊆ Train(500K))
                               + Frozen evaluation splits (dev, seen_test, unseen_test)
     4. Materialization      : Incremental original-byte copy -> Immutable TAR shards + receipts
     5. Verification Gate    : Checksums, schemas, quotas, zero cross-split leakage, inheritance
     6. Publication          : Metadata-last `release.json` commit on Google Drive (`cifaker`)
  -> Dataset Consumer (`load_release`)
     -> Verifies published release -> Stages/reuses verified TAR shards in Colab local cache
     -> ShardMapDataset / ShardIterableDataset -> Shared PyTorch DataLoader -> 4 CNN Pipelines
```

### Implemented Modules (`src/prooflens/data/`)

| Package / Module | Responsibility |
| :--- | :--- |
| `config.py` | Validated YAML schemas (`SourceConfig`, `ProtocolConfig`, `StorageConfig`, `ReleaseConfig`), strict unknown-key rejection, and deterministic SHA-256 `config_hash`. |
| `sources/base.py` | `SourceAdapter` contract, explicit `SourceCapabilities`, `SourceRecord`, `SourceImagePayload`. |
| `sources/archive.py` | `ArchiveSourceAdapter` for ZIP / TAR / TAR.GZ GenImage distributions using shard-aware single-pass streaming (`iter_selected_images`). |
| `sources/genimage.py` | `DirectoryGenImageAdapter`, revision-pinned `HuggingFaceGenImageAdapter` (requires explicit `hf_column_mapping`), and `create_source_adapter()`. |
| `registry/schema.py` | Canonical image schema, Parquet I/O (`pyarrow`), path safety, provenance-derived `image_id`, and strict REAL generator provenance hygiene (`generator="none"` for REAL images vs `source_generator_group`). |
| `registry/inventory.py` | Streaming batch `ParquetWriter` inventory builder (`O(batch_size)` memory). |
| `registry/duplicates.py` | Exact SHA-256, optional RGB `pixel_hash`, and Multi-Index Band LSH 64-bit `dHash` near-duplicate candidate generator + Union-Find leakage groups. |
| `sampling/leakage_guard.py` | Split boundary leakage validator & upstream source-split leakage quarantine/fail guard. |
| `sampling/stratification.py` | Size-invariant SHA-256 keyed group ranking, balanced generator/class quota distribution, conflict policies, and format/resolution confounding diagnostics. |
| `sampling/planner.py` | Nested release planner, frozen evaluation splits, dry-run feasibility report, replay verification (`verify_plan_replay`), and explicit replacement planning for exclusions. |
| `processing/validator.py` | Single-pass raw image bytes validation (Pillow decode, dimensions, format, SHA-256, optional pixel hash & dHash) preserving original bytes untouched. |
| `processing/shard_writer.py` | Staged immutable uncompressed TAR shard writer with fixed deterministic headers, byte-offset indexing, SHA-256 verification, and per-shard receipts. |
| `processing/materializer.py` | Bounded-memory incremental materialization with parent shard reuse, per-shard receipt resume, bounded concurrency, retries, and explicit `exclusions.jsonl`. |
| `storage/base.py`, `local.py`, `google_drive.py` | Replaceable storage abstraction with `LocalFilesystemStorage` and `GoogleDriveStorage` (`api` mode with MD5/size upload verification + `mounted` mode with fsync & read-back SHA-256 verification). |
| `releases/state.py`, `verifier.py`, `manager.py` | Explicit release lifecycle (`PLANNED -> MATERIALIZING -> MATERIALIZED -> VERIFIED -> PUBLISHED`, plus `FAILED`/`INTERRUPTED`), pre-publish verifier, metadata-last `ReleaseManager`, and read-only `ReleaseConsumer` (`FullLocalShardCache`, `BoundedShardCache`, `load_release`). |
| `shard_dataset.py` | `ShardMapDataset` (O(1) TAR offset seek) and `ShardIterableDataset` (bounded-disk streaming with disjoint multi-worker sharding and epoch length). |
| `benchmark.py` | Lightweight stage latency, throughput (img/s, MB/s), and `tracemalloc` peak memory profiler. |
| `cli.py` | Unified CLI: `python -m prooflens.data.cli {inventory,plan,build,verify,publish,prepare,benchmark}`. |

---

## 2. Canonical Data Contracts

### Canonical Image Schema (`manifest.parquet`)
- **Required logical fields:** `image_id`, `source_id`, `source_revision`, `source_path`, `source_split`, `label`, `generator`, `source_generator_group`, `source_shard`, `imagenet_class`, `leakage_group_id`, `eligibility_status`.
- **Optional metadata fields:** `width`, `height`, `image_format`, `file_size_bytes`, `sha256`, `pixel_hash`, `perceptual_hash`, `decode_status`, `exclusion_reason`.
- **Release shard columns (in `manifest.parquet`):** `release_split`, `shard_id`, `shard_storage_key`, `shard_member_name`, `shard_offset_data`, `origin_release_id`.
- **REAL provenance hygiene:** `REAL = 0`, `FAKE = 1`. REAL images MUST carry `generator ∈ {"none", "real", "natural"}` so they are never falsely attributed to a generative model; the paired GenImage subset folder (e.g., `biggan`, `sdv5`) is recorded separately in `source_generator_group`.

### Published Release Layout (in Storage)
```text
releases/{release_id}/
├── state.json                  # Explicit lifecycle state history
├── plan.json                   # Frozen release plan summary & config_hash
├── plan_selection.parquet      # Frozen selected image_ids & release_split
├── shards/{split}/
│   ├── {shard_id}.tar          # Immutable uncompressed TAR (original image bytes)
│   └── {shard_id}.receipt.json # Verified per-shard receipt & member byte offsets
├── manifest.parquet            # Final validated release manifest
├── checksums.json              # SHA-256 inventory of manifest and all referenced shards
├── audit_summary.json          # Resolved config, source provenance, duplicate & confounding audit
└── release.json                # WRITTEN LAST — presence + status=="PUBLISHED" gates consumers
```

---

## 3. Configuration Organization (`configs/`)

- `configs/sources/`: `genimage_full_archives.yaml`, `genimage_directory.yaml`
- `configs/protocols/`: `cross_generator_v1.yaml` (6 seen generators, 2 unseen generators, frozen 40K eval suite)
- `configs/storage/`: `local.yaml`, `google_drive_api.yaml` (folder `cifaker`, ID `1IipfeWX_OVJS2KwaFL5wFlO-edxLoEGD`), `google_drive_mounted.yaml`
- `configs/releases/`: `genimage_120k_v1.yaml`, `genimage_200k_v1.yaml`, `genimage_500k_v1.yaml`

---

## 4. Exact CLI Commands

```bash
# 1. Inspect available commands
python -m prooflens.data.cli --help

# 2. Build bounded-memory Parquet inventory from upstream source
python -m prooflens.data.cli inventory --config configs/releases/genimage_120k_v1.yaml

# 3. Dry-run release planning & feasibility verification (no image downloads / uploads)
python -m prooflens.data.cli plan --config configs/releases/genimage_120k_v1.yaml --dry-run

# 4. Materialize immutable TAR shards (resumable & idempotent)
python -m prooflens.data.cli build --config configs/releases/genimage_120k_v1.yaml --resume

# 5. Verify checksums, TAR structure, quotas, leakage, and parent inheritance
python -m prooflens.data.cli verify --release genimage-120k-v1 --storage-config configs/storage/google_drive_api.yaml

# 6. Publish immutable release.json last (only succeeds after verify passes)
python -m prooflens.data.cli publish --release genimage-120k-v1 --storage-config configs/storage/google_drive_api.yaml

# 7. Prepare a published split into local Colab cache (reuses verified cached shards)
python -m prooflens.data.cli prepare --release genimage-120k-v1 --split train --storage-config configs/storage/google_drive_api.yaml --cache-dir /content/prooflens_cache

# 8. Profile pipeline throughput and Python heap peak memory
python -m prooflens.data.cli benchmark --config configs/releases/genimage_120k_v1.yaml
```

---

## 5. Dataset Consumption in Model Training (Python API)

Model developers never need access to Full GenImage — only the codebase, Google Drive release access, release ID, and local cache directory:

```python
from torchvision import transforms
from prooflens.data import (
    create_shard_dataloader,
    create_storage_backend,
    load_release,
    load_storage_config,
)

storage = create_storage_backend(load_storage_config("configs/storage/google_drive_api.yaml"))

# Model-specific preprocessing stays in the training pipeline (never baked into canonical shards)
model_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

train_ds = load_release(
    release="genimage-120k-v1",
    split="train",
    cache_dir="/content/prooflens_cache",
    storage=storage,
    transform=model_transform,
)
train_loader = create_shard_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=4)
```

For larger releases (`genimage-500k-v1`) on disk-constrained runtimes, pass `streaming=True, max_cache_bytes=20 * 1024**3` to activate `BoundedShardCache` (LRU shard eviction) with `ShardIterableDataset`.

---

## 6. Scaling Incrementally: 120K → 200K → 500K

1. `genimage-200k-v1.yaml` sets `parent_release_id: "genimage-120k-v1"` and overrides `split_quotas.train` from 40K/40K to 80K/80K.
2. During `plan`, all frozen splits (`dev`, `seen_test`, `unseen_test`) are locked to the parent's exact IDs, and all 80K parent `train` IDs are included first before drawing the next 80K delta images from the exact same deterministic ranked stream (`Train(120K) ⊆ Train(200K) ⊆ Train(500K)`).
3. During `build`, every committed parent shard is verified and referenced directly (`origin_release_id: "genimage-120k-v1"`). Only the newly selected delta images are read from source and written into new TAR shards.
4. `genimage-500k-v1` follows the exact same chain with `parent_release_id: "genimage-200k-v1"`.

---

## 7. Local Tests vs. Opt-In Colab Real-Data Integration Test

- **Default local test suite (offline, CPU-only, zero credentials):**
  ```bash
  python -m pytest -q
  ```
- **Opt-in real GenImage integration test (run later on Google Colab once Full GenImage is mounted):**
  ```bash
  export PROOFLENS_REAL_GENIMAGE_ROOT="/content/genimage_archives"
  python -m pytest tests/data/test_shards_storage_and_e2e.py -k test_real_genimage_opt_in -s
  ```

---

## 8. Known Limitations & Cloud Integration Notes

- **Google Drive cloud execution is NOT verified locally:** `GoogleDriveStorage` (`api` and `mounted` modes) has been unit-tested offline using mounted filesystem emulation and an in-memory fake Drive v3 service. Real Colab OAuth authentication, Drive API quota/rate-limit behavior, and multi-gigabyte upload speeds must be verified during the first live Colab run.
- **No cross-file atomic transactions on Google Drive:** Publication relies on a single-writer policy and metadata-last commit (`release.json` uploaded last).
- **Duplicate audit scope:** Unless `--validate-bytes` is run on the full upstream inventory, exact SHA-256 and perceptual dHash duplicate auditing covers materialized release images only and does NOT prove the entire upstream source is duplicate-free.
- **Mid-epoch resume:** `ShardIterableDataset` supports deterministic epoch-level shuffling via `set_epoch(epoch)` and disjoint multi-worker sharding, but does NOT implement mid-epoch exact sample resume.
