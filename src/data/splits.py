"""
Builds standard and zero-shot splits from PrimeKG therapeutic edges.

Standard split:
  - Randomly hold out 10% of (drug, relation, disease) therapeutic edges as test.
  - Test diseases MAY have other treatment edges in training.
  - Val: 10% of remaining after test is removed.
  - Train: the rest.

Zero-shot split:
  - Select a held-out set of diseases that are ENTIRELY removed from training.
  - All therapeutic edges involving those diseases go to test.
  - No held-out disease has ANY treatment edge in train.
  - Val: 20% of held-out diseases used for validation.
  - Test: remaining 80% of held-out diseases.

Saves to:
  data/splits/standard/{train,val,test}.csv
  data/splits/zeroshot/{train,val,test}.csv
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
THERAPEUTIC_RELATIONS = {"indication", "contraindication"}

STANDARD_TEST_FRAC = 0.10
STANDARD_VAL_FRAC = 0.10
ZEROSHOT_HELD_OUT_FRAC = 0.20   # fraction of diseases held out entirely


def build_splits(kg: pd.DataFrame, seeds: list[int] = (42, 0, 1)):
    """
    Builds splits for all seeds. Each seed gets its own subdirectory.
    Canonical seed=42 split is what all model comparisons use.
    """
    therapeutic = kg[kg["relation"].isin(THERAPEUTIC_RELATIONS)].copy()
    print(f"[splits] Therapeutic edges: {len(therapeutic):,}")
    print(f"[splits] Unique diseases: {therapeutic['y_id'].nunique():,}")
    print(f"[splits] Unique drugs: {therapeutic['x_id'].nunique():,}")

    for seed in seeds:
        print(f"\n[splits] Building standard split (seed={seed})")
        _build_standard_split(therapeutic, seed)
        print(f"[splits] Building zero-shot split (seed={seed})")
        _build_zeroshot_split(therapeutic, seed, kg)

    print("\n[splits] Done.")


def _build_standard_split(therapeutic: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(therapeutic))
    rng.shuffle(idx)

    n_test = int(len(idx) * STANDARD_TEST_FRAC)
    n_val = int(len(idx) * STANDARD_VAL_FRAC)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    out_dir = SPLITS_DIR / "standard" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    therapeutic.iloc[train_idx].to_csv(out_dir / "train.csv", index=False)
    therapeutic.iloc[val_idx].to_csv(out_dir / "val.csv", index=False)
    therapeutic.iloc[test_idx].to_csv(out_dir / "test.csv", index=False)

    meta = {
        "split_type": "standard",
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "note": "Test diseases may have training treatment edges. Not zero-shot."
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")


def _build_zeroshot_split(therapeutic: pd.DataFrame, seed: int, full_kg: pd.DataFrame):
    """
    Held-out diseases have ZERO treatment edges in train.
    Remaining (non-therapeutic) edges from those diseases ARE allowed in train
    (they contribute structural information but no drug labels).
    """
    rng = np.random.default_rng(seed)
    all_diseases = therapeutic["y_id"].unique()
    all_diseases = all_diseases.to_numpy() if hasattr(all_diseases, "to_numpy") else all_diseases
    rng.shuffle(all_diseases)

    n_held = int(len(all_diseases) * ZEROSHOT_HELD_OUT_FRAC)
    held_out = set(all_diseases[:n_held])
    seen = set(all_diseases[n_held:])

    # Val = 20% of held-out diseases, Test = 80%
    held_list = list(held_out)
    n_val_diseases = max(1, int(n_held * 0.20))
    val_diseases = set(held_list[:n_val_diseases])
    test_diseases = set(held_list[n_val_diseases:])

    train_mask = therapeutic["y_id"].isin(seen)
    val_mask = therapeutic["y_id"].isin(val_diseases)
    test_mask = therapeutic["y_id"].isin(test_diseases)

    # Sanity: held-out diseases must have ZERO training edges
    assert not any(therapeutic[train_mask]["y_id"].isin(held_out)), \
        "BUG: held-out disease appears in training edges"

    out_dir = SPLITS_DIR / "zeroshot" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    therapeutic[train_mask].to_csv(out_dir / "train.csv", index=False)
    therapeutic[val_mask].to_csv(out_dir / "val.csv", index=False)
    therapeutic[test_mask].to_csv(out_dir / "test.csv", index=False)

    meta = {
        "split_type": "zeroshot",
        "seed": seed,
        "n_train_edges": int(train_mask.sum()),
        "n_val_edges": int(val_mask.sum()),
        "n_test_edges": int(test_mask.sum()),
        "n_seen_diseases": len(seen),
        "n_val_diseases": len(val_diseases),
        "n_test_diseases": len(test_diseases),
        "n_total_diseases": len(all_diseases),
        "held_out_frac": ZEROSHOT_HELD_OUT_FRAC,
        "note": (
            "Zero-shot: held-out diseases have no treatment edges in train. "
            "Non-therapeutic edges from held-out diseases are still in the full KG "
            "for structural context."
        )
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  train={train_mask.sum():,}  val={val_mask.sum():,}  test={test_mask.sum():,}")
    print(f"  seen_diseases={len(seen):,}  val_diseases={len(val_diseases):,}  test_diseases={len(test_diseases):,}")
