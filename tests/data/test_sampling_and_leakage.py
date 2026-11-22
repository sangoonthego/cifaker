"""Unit tests for deterministic nested sampling, leakage guards, and quota policies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from prooflens.data.config import (
    ProtocolConfig,
    ReleaseConfig,
    SourceConfig,
    SplitRuleConfig,
    StorageConfig,
)
from prooflens.data.registry.schema import LABEL_FAKE, LABEL_REAL, make_canonical_image_id
from prooflens.data.sampling.leakage_guard import (
    LeakageViolationError,
    check_leakage_boundaries,
)
from prooflens.data.sampling.planner import (
    plan_release,
    plan_replacements_for_exclusions,
    verify_plan_replay,
)
from prooflens.data.sampling.stratification import QuotaInfeasibleError


def _make_synthetic_inventory(
    seen_gens: tuple[str, ...] = ("biggan", "sdv5"),
    unseen_gens: tuple[str, ...] = ("midjourney",),
    train_per_gen_per_label: int = 20,
    val_per_gen_per_label: int = 10,
) -> pd.DataFrame:
    rows = []
    for gen in seen_gens + unseen_gens:
        for split, count in (("train", train_per_gen_per_label), ("val", val_per_gen_per_label)):
            for label in (LABEL_REAL, LABEL_FAKE):
                cls_dir = "nature" if label == LABEL_REAL else "ai"
                ext = ".jpg" if label == LABEL_REAL else ".png"
                for i in range(count):
                    path = f"{gen}/{split}/{cls_dir}/n01440{i % 3:03d}_{i:04d}{ext}"
                    img_id = make_canonical_image_id("synth_src", "v1", path)
                    rows.append(
                        {
                            "image_id": img_id,
                            "source_id": "synth_src",
                            "source_revision": "v1",
                            "source_path": path,
                            "source_split": split,
                            "label": label,
                            "generator": "none" if label == LABEL_REAL else gen,
                            "source_generator_group": gen,
                            "source_shard": gen,
                            "imagenet_class": f"n01440{i % 3:03d}",
                            "leakage_group_id": f"lg_{img_id}",
                            "eligibility_status": "ELIGIBLE",
                            "width": 64,
                            "height": 64,
                            "image_format": "JPEG" if label == LABEL_REAL else "PNG",
                            "file_size_bytes": 1024,
                            "sha256": f"{img_id}_sha",
                        }
                    )
    return pd.DataFrame(rows)


def _make_release_cfg(
    release_id: str,
    train_quota_each: int,
    parent_id: str | None = None,
    leakage_policy: str = "fail",
    group_quota_policy: str = "exact_or_fail",
    deduplicate_within_group: bool = True,
    seed: int = 1234,
) -> ReleaseConfig:
    return ReleaseConfig(
        release_id=release_id,
        sampling_seed=seed,
        parent_release_id=parent_id,
        source=SourceConfig(
            source_id="synth_src",
            source_type="directory",
            source_revision="v1",
            location="unused",
        ),
        protocol=ProtocolConfig(
            protocol_version="cross_gen_test_v1",
            seen_generators=("biggan", "sdv5"),
            unseen_generators=("midjourney",),
            group_quota_policy=group_quota_policy,
            cross_source_split_leakage_policy=leakage_policy,
            deduplicate_within_group=deduplicate_within_group,
            splits={
                "train": SplitRuleConfig(
                    allowed_source_splits=("train",),
                    generator_pool="seen",
                    frozen_across_scaling=False,
                    real_quota=train_quota_each,
                    fake_quota=train_quota_each,
                ),
                "dev": SplitRuleConfig(
                    allowed_source_splits=("train",),
                    generator_pool="seen",
                    frozen_across_scaling=True,
                    real_quota=4,
                    fake_quota=4,
                ),
                "seen_test": SplitRuleConfig(
                    allowed_source_splits=("val",),
                    generator_pool="seen",
                    frozen_across_scaling=True,
                    real_quota=4,
                    fake_quota=4,
                ),
                "unseen_test": SplitRuleConfig(
                    allowed_source_splits=("val",),
                    generator_pool="unseen",
                    frozen_across_scaling=True,
                    real_quota=4,
                    fake_quota=4,
                ),
            },
        ),
        storage=StorageConfig(backend="local", root_path="unused"),
    )


def test_deterministic_nested_sampling_and_replay(tmp_path: Path) -> None:
    inv = _make_synthetic_inventory()

    # Shuffling inventory row order must NOT change planned selection
    inv_shuffled = inv.sample(frac=1.0, random_state=777).reset_index(drop=True)

    cfg_small = _make_release_cfg("rel-small-v1", train_quota_each=8)
    cfg_med = _make_release_cfg("rel-med-v1", train_quota_each=16)
    cfg_large = _make_release_cfg("rel-large-v1", train_quota_each=24)

    p_small = plan_release(inv, cfg_small)
    p_small_shuf = plan_release(inv_shuffled, cfg_small)
    p_med = plan_release(inv, cfg_med)
    p_large = plan_release(inv, cfg_large)

    assert p_small.plan_id == p_small_shuf.plan_id

    # Verify exact replay from disk
    _, sel_pq = p_small.save(tmp_path / "plan_small")
    assert verify_plan_replay(inv_shuffled, cfg_small, sel_pq) is True

    def _split_ids(df: pd.DataFrame, split: str) -> set[str]:
        return set(df.loc[df["release_split"] == split, "image_id"].tolist())

    # Nested training invariant: Train(small) ⊆ Train(med) ⊆ Train(large)
    assert _split_ids(p_small.selection_df, "train") < _split_ids(p_med.selection_df, "train")
    assert _split_ids(p_med.selection_df, "train") < _split_ids(p_large.selection_df, "train")

    # Frozen evaluation splits remain 100% identical across all scaling releases
    for eval_split in ("dev", "seen_test", "unseen_test"):
        assert _split_ids(p_small.selection_df, eval_split) == _split_ids(
            p_med.selection_df, eval_split
        )
        assert _split_ids(p_med.selection_df, eval_split) == _split_ids(
            p_large.selection_df, eval_split
        )

    # Strict generator isolation & source split preservation
    td_df = p_large.selection_df[p_large.selection_df["release_split"].isin(["train", "dev"])]
    assert set(td_df["source_generator_group"]).issubset({"biggan", "sdv5"})
    assert set(td_df["source_split"]) == {"train"}

    unseen_df = p_large.selection_df[p_large.selection_df["release_split"] == "unseen_test"]
    assert set(unseen_df["source_generator_group"]) == {"midjourney"}
    assert set(unseen_df["source_split"]) == {"val"}

    # Exact class balance per split
    for sp, grp in p_large.selection_df.groupby("release_split"):
        assert (grp["label"] == LABEL_REAL).sum() == (grp["label"] == LABEL_FAKE).sum(), sp

    # Confounding detection flags REAL=JPEG vs FAKE=PNG
    assert p_small.summary["confounding_risks"]["format_confounding_detected"] is True


def test_cross_source_split_leakage_rejection_and_quarantine() -> None:
    inv = _make_synthetic_inventory()
    # Inject identical leakage group across upstream train and val
    train_idx = inv[(inv["source_split"] == "train") & (inv["label"] == 1)].index[0]
    val_idx = inv[(inv["source_split"] == "val") & (inv["label"] == 1)].index[0]
    inv.loc[val_idx, "leakage_group_id"] = inv.loc[train_idx, "leakage_group_id"]
    inv.loc[val_idx, "sha256"] = inv.loc[train_idx, "sha256"]

    # Policy="fail" must reject explicitly
    cfg_fail = _make_release_cfg("rel-leak-fail", train_quota_each=6, leakage_policy="fail")
    with pytest.raises(LeakageViolationError, match="Upstream source_split leakage detected"):
        plan_release(inv, cfg_fail)

    # Policy="exclude_cross_split_groups" quarantines leaking items and ensures clean selection
    cfg_quar = _make_release_cfg(
        "rel-leak-quar", train_quota_each=6, leakage_policy="exclude_cross_split_groups"
    )
    plan = plan_release(inv, cfg_quar)
    assert plan.feasible is True
    leaking_ids = {inv.loc[train_idx, "image_id"], inv.loc[val_idx, "image_id"]}
    assert leaking_ids.isdisjoint(set(plan.selection_df["image_id"]))
    assert check_leakage_boundaries(plan.selection_df, split_column="release_split").passed is True


def test_infeasible_quota_and_group_conflict_policies() -> None:
    inv = _make_synthetic_inventory(train_per_gen_per_label=3, val_per_gen_per_label=4)
    # Ask for more train samples than exist (2 seen gens * 3 = 6 available, asking 10 + 4 dev = 14)
    cfg_too_big = _make_release_cfg("rel-infeasible", train_quota_each=10)
    with pytest.raises(QuotaInfeasibleError, match="Infeasible quota"):
        plan_release(inv, cfg_too_big)

    # Group conflict test: put every train image into 3-item leakage groups with deduplicate_within_group=False
    # Requesting 2 per generator cannot be met exactly by 3-item groups without splitting a group!
    inv_grouped = _make_synthetic_inventory(train_per_gen_per_label=9, val_per_gen_per_label=6)
    mask_train = inv_grouped["source_split"] == "train"
    for (_gen, _lbl), grp in inv_grouped[mask_train].groupby(["source_generator_group", "label"]):
        idxs = list(grp.index)
        for pos, idx in enumerate(idxs):
            inv_grouped.loc[idx, "leakage_group_id"] = f"triple_{_gen}_{_lbl}_{pos // 3}"

    cfg_strict = _make_release_cfg(
        "rel-grp-strict",
        train_quota_each=4,  # 2 per generator -> conflicts with group size 3!
        group_quota_policy="exact_or_fail",
        deduplicate_within_group=False,
    )
    with pytest.raises(QuotaInfeasibleError):
        plan_release(inv_grouped, cfg_strict)

    cfg_underfill = _make_release_cfg(
        "rel-grp-underfill",
        train_quota_each=4,
        group_quota_policy="allow_underfill",
        deduplicate_within_group=False,
    )
    under_plan = plan_release(inv_grouped, cfg_underfill)
    assert under_plan.feasible is False
    assert len(under_plan.summary["missing_quota_report"]) > 0
    assert check_leakage_boundaries(under_plan.selection_df, split_column="release_split").passed


def test_deterministic_replacement_plan_after_exclusions() -> None:
    inv = _make_synthetic_inventory()
    cfg = _make_release_cfg("rel-replace-v1", train_quota_each=6)
    p1 = plan_release(inv, cfg)

    corrupted_id = str(p1.selection_df.iloc[0]["image_id"])
    p2 = plan_replacements_for_exclusions(
        inv, cfg, excluded_image_ids={corrupted_id}, exclusion_reasons={corrupted_id: "CORRUPTED"}
    )
    assert p2.feasible is True
    assert corrupted_id not in set(p2.selection_df["image_id"])
    assert p2.plan_id != p1.plan_id
    assert len(p2.selection_df) == len(p1.selection_df)
