"""Duplicate detection and versioned leakage-group builder for ProofLens.

Detection layers (strictly separated):
1. Exact file-byte duplicates via SHA-256 (`sha256`).
2. Optional decoded RGB pixel-buffer duplicates via `pixel_hash` (catches lossless
   PNG vs BMP re-encoding of identical pixels).
3. Optional near-duplicate candidate generation via 64-bit difference hash (`dHash`
   in `perceptual_hash`) using Multi-Index Band LSH (4 × 16-bit bands) to avoid
   O(N²) exhaustive pairwise comparisons.

Policy hygiene:
- Perceptual similarity is NEVER treated as definitive duplication unless the
  caller explicitly enables `include_perceptual_in_leakage_groups=True`.
- Every audit report explicitly declares its `audit_scope` so auditing a selected
  subset is never misconstrued as proving the entire upstream dataset is duplicate-free.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd


class _UnionFind:
    """Disjoint-set data structure with path compression and union by rank."""

    def __init__(self, items: list[str]) -> None:
        self.parent: dict[str, str] = {x: x for x in items}
        self.rank: dict[str, int] = dict.fromkeys(items, 0)

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def hamming_distance_hex64(hex_a: str, hex_b: str) -> int:
    """Compute bit-level Hamming distance between two 64-bit hex perceptual hashes."""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def find_perceptual_near_duplicate_candidates(
    image_ids: list[str],
    perceptual_hashes: list[str | None],
    max_hamming_distance: int = 4,
    num_bands: int = 4,
    max_bucket_size: int = 256,
) -> list[dict[str, Any]]:
    """Find near-duplicate candidate pairs using Multi-Index Band LSH (avoids O(N²) scans).

    A 64-bit hex hash (16 hex chars) is split into `num_bands` equal chunks (default:
    4 bands × 16 bits = 4 hex chars). By the pigeonhole principle, any two hashes
    within Hamming distance < `num_bands` are guaranteed to share at least one band
    exactly; with `max_hamming_distance=4` and 4 bands, recall for distance <= 3 is 100%
    and remains very high for distance 4 while scanning only local band buckets.
    """
    if max_hamming_distance < 0 or max_hamming_distance > 64:
        raise ValueError(f"max_hamming_distance must be in [0, 64], got {max_hamming_distance}")
    if 16 % num_bands != 0:
        raise ValueError(f"num_bands must divide 16 hex chars cleanly, got {num_bands}")

    band_width = 16 // num_bands
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    valid_entries: list[tuple[str, str]] = []

    for img_id, ph in zip(image_ids, perceptual_hashes, strict=True):
        if not ph or not isinstance(ph, str):
            continue
        clean = ph.strip().lower()
        if len(clean) != 16:
            continue
        idx = len(valid_entries)
        valid_entries.append((img_id, clean))
        for b in range(num_bands):
            slice_key = clean[b * band_width : (b + 1) * band_width]
            buckets[(b, slice_key)].append(idx)

    seen_pairs: set[tuple[int, int]] = set()
    candidates: list[dict[str, Any]] = []

    for indices in buckets.values():
        if len(indices) < 2:
            continue
        # Guard against degenerate uniform-image buckets blowing up quadratically
        capped = indices[:max_bucket_size]
        n = len(capped)
        for i in range(n):
            idx_a = capped[i]
            id_a, hash_a = valid_entries[idx_a]
            for j in range(i + 1, n):
                idx_b = capped[j]
                pair = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                id_b, hash_b = valid_entries[idx_b]
                dist = hamming_distance_hex64(hash_a, hash_b)
                if dist <= max_hamming_distance:
                    a_id, b_id = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                    candidates.append(
                        {
                            "image_id_a": a_id,
                            "image_id_b": b_id,
                            "hamming_distance": dist,
                        }
                    )

    candidates.sort(key=lambda c: (c["hamming_distance"], c["image_id_a"], c["image_id_b"]))
    return candidates


@dataclass(frozen=True)
class DuplicateAuditReport:
    """Structured result of running exact, pixel, and perceptual duplicate analysis."""

    audit_scope: str
    total_records_inspected: int
    records_with_sha256: int
    exact_sha256_duplicate_groups: int
    exact_sha256_duplicate_files: int
    cross_source_split_sha256_groups: int
    cross_label_sha256_groups: int
    pixel_hash_duplicate_groups: int
    perceptual_candidate_pairs: int
    perceptual_included_in_leakage_groups: bool
    leakage_group_count: int
    multi_member_leakage_groups: int
    sample_exact_groups: list[dict[str, Any]]
    sample_near_duplicate_candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_scope": self.audit_scope,
            "total_records_inspected": self.total_records_inspected,
            "records_with_sha256": self.records_with_sha256,
            "exact_sha256_duplicate_groups": self.exact_sha256_duplicate_groups,
            "exact_sha256_duplicate_files": self.exact_sha256_duplicate_files,
            "cross_source_split_sha256_groups": self.cross_source_split_sha256_groups,
            "cross_label_sha256_groups": self.cross_label_sha256_groups,
            "pixel_hash_duplicate_groups": self.pixel_hash_duplicate_groups,
            "perceptual_candidate_pairs": self.perceptual_candidate_pairs,
            "perceptual_included_in_leakage_groups": self.perceptual_included_in_leakage_groups,
            "leakage_group_count": self.leakage_group_count,
            "multi_member_leakage_groups": self.multi_member_leakage_groups,
            "sample_exact_groups": self.sample_exact_groups,
            "sample_near_duplicate_candidates": self.sample_near_duplicate_candidates,
        }


def assign_leakage_groups(
    df: pd.DataFrame,
    audit_scope: str = "selected_or_materialized_subset",
    include_pixel_hash: bool = True,
    include_perceptual_in_leakage_groups: bool = False,
    perceptual_hamming_threshold: int = 4,
) -> tuple[pd.DataFrame, DuplicateAuditReport]:
    """Compute deterministic `leakage_group_id` clusters via connected components.

    Merges records sharing:
    - identical existing non-trivial `leakage_group_id` (if pre-assigned)
    - identical `sha256` (exact file-byte duplicates)
    - identical `pixel_hash` (if `include_pixel_hash=True`)
    - near-duplicate `perceptual_hash` pairs ONLY when `include_perceptual_in_leakage_groups=True`

    Returns:
        `(updated_df, DuplicateAuditReport)` with deterministic `leakage_group_id`s derived
        from the lexicographically smallest member `image_id` in each connected component.
    """
    if df.empty:
        empty_report = DuplicateAuditReport(
            audit_scope=audit_scope,
            total_records_inspected=0,
            records_with_sha256=0,
            exact_sha256_duplicate_groups=0,
            exact_sha256_duplicate_files=0,
            cross_source_split_sha256_groups=0,
            cross_label_sha256_groups=0,
            pixel_hash_duplicate_groups=0,
            perceptual_candidate_pairs=0,
            perceptual_included_in_leakage_groups=include_perceptual_in_leakage_groups,
            leakage_group_count=0,
            multi_member_leakage_groups=0,
            sample_exact_groups=[],
            sample_near_duplicate_candidates=[],
        )
        return df.copy(), empty_report

    out = df.copy()
    image_ids = out["image_id"].astype(str).tolist()
    uf = _UnionFind(image_ids)

    # Preserve pre-existing explicit leakage groupings
    if "leakage_group_id" in out.columns:
        by_existing: dict[str, list[str]] = defaultdict(list)
        for img_id, lg in zip(image_ids, out["leakage_group_id"].tolist(), strict=True):
            if lg and isinstance(lg, str):
                by_existing[lg].append(img_id)
        for members in by_existing.values():
            if len(members) > 1:
                head = members[0]
                for m in members[1:]:
                    uf.union(head, m)

    # 1. Exact SHA-256 duplicates
    sha_col = out["sha256"].tolist() if "sha256" in out.columns else [None] * len(out)
    splits_col = out["source_split"].astype(str).tolist()
    labels_col = out["label"].astype(int).tolist()
    by_sha: dict[str, list[int]] = defaultdict(list)
    records_with_sha = 0
    for idx, sha in enumerate(sha_col):
        if sha and isinstance(sha, str) and sha.strip():
            records_with_sha += 1
            by_sha[sha.strip()].append(idx)

    exact_dup_groups = 0
    exact_dup_files = 0
    cross_src_split_groups = 0
    cross_label_groups = 0
    sample_exact_groups: list[dict[str, Any]] = []

    for sha, indices in sorted(by_sha.items()):
        if len(indices) <= 1:
            continue
        exact_dup_groups += 1
        exact_dup_files += len(indices)
        head_id = image_ids[indices[0]]
        for i in indices[1:]:
            uf.union(head_id, image_ids[i])

        grp_splits = sorted({splits_col[i] for i in indices})
        grp_labels = sorted({labels_col[i] for i in indices})
        if len(grp_splits) > 1:
            cross_src_split_groups += 1
        if len(grp_labels) > 1:
            cross_label_groups += 1
        if len(sample_exact_groups) < 20:
            sample_exact_groups.append(
                {
                    "sha256": sha,
                    "count": len(indices),
                    "source_splits": grp_splits,
                    "labels": grp_labels,
                    "image_ids": sorted(image_ids[i] for i in indices)[:10],
                }
            )

    # 2. Optional pixel_hash duplicates
    pixel_dup_groups = 0
    if include_pixel_hash and "pixel_hash" in out.columns:
        by_px: dict[str, list[int]] = defaultdict(list)
        for idx, px in enumerate(out["pixel_hash"].tolist()):
            if px and isinstance(px, str) and px.strip():
                by_px[px.strip()].append(idx)
        for indices in by_px.values():
            if len(indices) > 1:
                pixel_dup_groups += 1
                head_id = image_ids[indices[0]]
                for i in indices[1:]:
                    uf.union(head_id, image_ids[i])

    # 3. Optional perceptual near-duplicate candidate generation via Band LSH
    near_candidates: list[dict[str, Any]] = []
    if "perceptual_hash" in out.columns:
        ph_list = out["perceptual_hash"].tolist()
        if any(isinstance(x, str) and len(x.strip()) == 16 for x in ph_list):
            near_candidates = find_perceptual_near_duplicate_candidates(
                image_ids=image_ids,
                perceptual_hashes=ph_list,
                max_hamming_distance=perceptual_hamming_threshold,
            )
            if include_perceptual_in_leakage_groups:
                for cand in near_candidates:
                    uf.union(cand["image_id_a"], cand["image_id_b"])

    # Assign deterministic group IDs based on canonical sorted member IDs
    components: dict[str, list[str]] = defaultdict(list)
    for img_id in image_ids:
        components[uf.find(img_id)].append(img_id)

    id_to_group: dict[str, str] = {}
    multi_member_groups = 0
    for members in components.values():
        members_sorted = sorted(members)
        if len(members_sorted) > 1:
            multi_member_groups += 1
            digest = hashlib.sha256("|".join(members_sorted).encode("utf-8")).hexdigest()[:16]
            lg_id = f"lg_grp_{digest}"
        else:
            lg_id = f"lg_{members_sorted[0]}"
        for m in members_sorted:
            id_to_group[m] = lg_id

    out["leakage_group_id"] = [id_to_group[img_id] for img_id in image_ids]

    report = DuplicateAuditReport(
        audit_scope=audit_scope,
        total_records_inspected=len(out),
        records_with_sha256=records_with_sha,
        exact_sha256_duplicate_groups=exact_dup_groups,
        exact_sha256_duplicate_files=exact_dup_files,
        cross_source_split_sha256_groups=cross_src_split_groups,
        cross_label_sha256_groups=cross_label_groups,
        pixel_hash_duplicate_groups=pixel_dup_groups,
        perceptual_candidate_pairs=len(near_candidates),
        perceptual_included_in_leakage_groups=include_perceptual_in_leakage_groups,
        leakage_group_count=len(components),
        multi_member_leakage_groups=multi_member_groups,
        sample_exact_groups=sample_exact_groups,
        sample_near_duplicate_candidates=near_candidates[:20],
    )
    return out, report
