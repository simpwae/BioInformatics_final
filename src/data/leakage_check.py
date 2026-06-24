"""
Verifies zero-shot split integrity: no held-out disease appears in any
training treatment edge. Writes results/metrics/leakage_check_{seed}.json.
Exit code 0 = PASS. Exit code 1 = FAIL.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
METRICS_DIR = ROOT / "results" / "metrics"
THERAPEUTIC_RELATIONS = {"indication", "contraindication"}


def check_leakage(seed: int = 42) -> dict:
    split_dir = SPLITS_DIR / "zeroshot" / f"seed_{seed}"
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"
    val_path = split_dir / "val.csv"

    for p in [train_path, test_path, val_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run: python scripts/download_primekg.py && "
                "python -c \"from src.data.splits import build_splits; ...\" first."
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    val = pd.read_csv(val_path)

    # Only check therapeutic edges
    train_t = train[train["relation"].isin(THERAPEUTIC_RELATIONS)]
    train_diseases = set(train_t["y_id"].unique())

    test_diseases = set(test["y_id"].unique())
    val_diseases = set(val["y_id"].unique())
    held_out_diseases = test_diseases | val_diseases

    leaking = sorted(str(d) for d in (held_out_diseases & train_diseases))

    result = {
        "status": "PASS" if len(leaking) == 0 else "FAIL",
        "seed": seed,
        "leaking_diseases": leaking,
        "n_train_diseases": len(train_diseases),
        "n_test_diseases": len(test_diseases),
        "n_val_diseases": len(val_diseases),
        "n_leaking": len(leaking),
        "checked_at": datetime.utcnow().isoformat(),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"leakage_check_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    result = check_leakage(seed)
    print(json.dumps(result, indent=2))
    if result["status"] == "FAIL":
        print(f"\n[FAIL] Leakage in seed={seed}. Fix the split before using zero-shot results.")
        sys.exit(1)
    print(f"\n[PASS] No leakage for seed={seed}.")
