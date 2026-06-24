"""Builds all splits then runs leakage check for all seeds. One-shot setup."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data.splits import build_splits
from src.data.leakage_check import check_leakage
import json

print("Loading kg.csv ...")
kg = pd.read_csv(ROOT / "data" / "raw" / "kg.csv", low_memory=False)
print(f"Loaded {len(kg):,} edges.")

print("\nBuilding splits for seeds [42, 0, 1] ...")
build_splits(kg, seeds=[42, 0, 1])

print("\nRunning leakage checks ...")
all_pass = True
for seed in [42, 0, 1]:
    result = check_leakage(seed)
    status = result["status"]
    print(f"  seed={seed}: {status} | n_test_diseases={result['n_test_diseases']} | n_leaking={result['n_leaking']}")
    if status == "FAIL":
        all_pass = False

if all_pass:
    print("\n[PASS] All splits clean. Zero-shot results are valid to use.")
else:
    print("\n[FAIL] Leakage detected. Do not use zero-shot results. Fix splits.")
    sys.exit(1)
