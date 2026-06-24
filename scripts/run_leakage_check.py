"""
Phase 9 — Leakage Check (gates all zero-shot results).

Reads the zero-shot split from data/splits/ and asserts that no disease
in the test set appears in any treatment edge in the training set.
Writes results/metrics/leakage_check.json.

EXIT CODE 0 = PASS. EXIT CODE 1 = FAIL (do not use zero-shot results).
"""

import json
import sys
import os
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(ROOT, "data", "splits")
OUTPUT = os.path.join(ROOT, "results", "metrics", "leakage_check.json")


def run_leakage_check():
    train_path = os.path.join(SPLITS_DIR, "zero_shot_train.csv")
    test_path = os.path.join(SPLITS_DIR, "zero_shot_test.csv")

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Split file not found: {p}")
            print("Download PrimeKG and run the split script first.")
            sys.exit(1)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Expected columns: disease_id, drug_id, relation (indication/contraindication)
    treatment_relations = {"indication", "contraindication"}

    train_treatment = train[train["relation"].isin(treatment_relations)]
    train_diseases = set(train_treatment["disease_id"].unique())

    test_diseases = set(test["disease_id"].unique())
    leaking = sorted(test_diseases & train_diseases)

    result = {
        "status": "PASS" if len(leaking) == 0 else "FAIL",
        "leaking_diseases": leaking,
        "n_test_diseases": len(test_diseases),
        "n_leaking": len(leaking),
        "checked_at": datetime.utcnow().isoformat()
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

    if result["status"] == "FAIL":
        print("\n[FAIL] Leakage detected. Fix the split before using zero-shot results.")
        sys.exit(1)

    print("\n[PASS] No leakage. Zero-shot results are valid to use.")


if __name__ == "__main__":
    run_leakage_check()
