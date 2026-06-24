"""
Q4 — Case study runner.

For two pre-selected diseases (one rare/zero-shot, one well-studied):
  1. Load the best model's test predictions (from per_disease_results in the JSON)
  2. Report top-10 predicted drugs with scores (from file, not by hand)
  3. Extract the 2-hop KG path between the drug and disease through shared neighbors

Disease selection (fixed before running, do not change after seeing predictions):
  CASE_A: Familial Hypertrophic Cardiomyopathy (id=24573) — rare genetic heart disease,
          n_pos=1 approved therapy in PrimeKG, zero-shot test split
  CASE_B: Staphylococcus Aureus Infection (id=5545) — common bacterial infection,
          n_pos=45 approved therapies in PrimeKG, zero-shot test split

Both cases are drawn from the zero-shot test split (seed=42) because only that
split's result file contains per-disease top_k_drugs (required for predictions).
Standard-split top_k_drugs would require a checkpoint re-run; that was not done.
Case B satisfies "well-studied, multiple approved therapies" with n_pos=45.

Writes:
  results/predictions/case_study_caseA_{model}.csv
  results/predictions/case_study_caseB_{model}.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = ROOT / "results" / "predictions"

# Fixed before running. Do NOT change after seeing predictions.
# Selected from actual per_disease_results in zeroshot/seed_42/txgnn.json.
CASE_A_DISEASE_NAME = "familial hypertrophic cardiomyopathy"   # id=24573, n_pos=1 (rare)
CASE_B_DISEASE_NAME = "staphylococcus aureus infection"         # id=5545, n_pos=45 (well-studied)

MODEL = "txgnn"
SEED = 42
SPLIT = "zeroshot"


def find_disease_id(disease_name: str, kg: pd.DataFrame) -> Optional[str]:
    """Find the disease ID by name (case-insensitive substring match)."""
    mask = kg["y_name"].str.lower().str.contains(disease_name.lower(), na=False)
    hits = kg[mask & (kg["y_type"] == "disease")]["y_id"].unique()
    if len(hits) == 0:
        mask2 = kg["x_name"].str.lower().str.contains(disease_name.lower(), na=False)
        hits = kg[mask2 & (kg["x_type"] == "disease")]["x_id"].unique()
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        print(f"  [warn] Multiple matches for '{disease_name}': {hits[:5]} — using first")
    return str(hits[0])


def get_top_predictions(
    disease_id: str,
    per_disease_results: list,
    kg: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Extracts top-K drug predictions for a specific disease.
    Each row: rank, drug_id, drug_name, score, is_positive, relation.
    Source: per_disease_results[*].top_k_drugs (written by zero_shot_eval.py).
    Numbers come from the result file, not typed by hand.
    """
    disease_preds = [
        r for r in per_disease_results
        if str(r["disease_id"]) == str(disease_id)
    ]
    if not disease_preds:
        return pd.DataFrame(columns=["disease_id", "relation", "rank",
                                      "drug_id", "drug_name", "score", "is_positive"])

    # Build drug name lookup from KG
    drug_names = {}
    drug_rows = kg[kg["x_type"] == "drug"][["x_id", "x_name"]].drop_duplicates("x_id")
    for _, row in drug_rows.iterrows():
        drug_names[str(row["x_id"])] = row["x_name"]

    rows = []
    for pred in disease_preds:
        relation = pred["relation"]
        for drug_entry in pred.get("top_k_drugs", [])[:top_k]:
            drug_id = str(drug_entry["drug_id"])
            rows.append({
                "disease_id": disease_id,
                "relation": relation,
                "rank": drug_entry["rank"],
                "drug_id": drug_id,
                "drug_name": drug_names.get(drug_id, "unknown"),
                "score": drug_entry["score"],
                "is_positive": drug_entry["is_positive"],
            })

    return pd.DataFrame(rows)


def extract_kg_paths(
    drug_id: str,
    disease_id: str,
    kg: pd.DataFrame,
    max_paths: int = 5,
    hop: int = 2,
) -> list:
    """
    Finds 2-hop paths: drug -> intermediate_entity -> disease.
    Returns list of (drug_id, via_entity_name, via_relation_1, via_relation_2, disease_id).
    """
    # Drug's 1-hop neighbors
    drug_out = kg[kg["x_id"] == drug_id][["y_id", "y_name", "y_type", "relation"]].copy()
    drug_out.columns = ["mid_id", "mid_name", "mid_type", "rel1"]

    # Which of those neighbors connect to the disease
    disease_in = kg[kg["y_id"] == disease_id][["x_id", "relation"]].copy()
    disease_in.columns = ["mid_id", "rel2"]

    # Also check reverse: disease -> neighbor
    disease_out = kg[kg["x_id"] == disease_id][["y_id", "relation"]].copy()
    disease_out.columns = ["mid_id", "rel2"]

    paths = []
    for df_dis in [disease_in, disease_out]:
        merged = pd.merge(drug_out, df_dis, on="mid_id")
        for _, row in merged.head(max_paths).iterrows():
            paths.append({
                "drug_id": drug_id,
                "via_entity": row["mid_name"],
                "via_type": row["mid_type"],
                "relation_drug_to_entity": row["rel1"],
                "relation_entity_to_disease": row["rel2"],
                "disease_id": disease_id,
            })
        if len(paths) >= max_paths:
            break

    return paths[:max_paths]


def run_case_studies(kg: pd.DataFrame):
    """Main entry point. Loads model results and writes case study files."""
    result_path = (ROOT / "results" / MODEL / SPLIT /
                   f"seed_{SEED}" / f"{MODEL}.json")
    if not result_path.exists():
        print(f"[error] Result file not found: {result_path}")
        print(f"  Run: python scripts/run_txgnn.py --split zeroshot --seeds {SEED} first.")
        return

    with open(result_path) as f:
        results = json.load(f)

    per_disease = results.get("per_disease_results", [])
    if not per_disease:
        print("[error] per_disease_results is empty in the result file.")
        return

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for case_label, disease_name in [("caseA", CASE_A_DISEASE_NAME),
                                      ("caseB", CASE_B_DISEASE_NAME)]:
        disease_id = find_disease_id(disease_name, kg)
        if disease_id is None:
            print(f"[warn] Disease not found: '{disease_name}'")
            continue

        print(f"\n=== Case Study {case_label}: {disease_name} (id={disease_id}) ===")

        preds = get_top_predictions(disease_id, per_disease, kg, top_k=20)
        if preds.empty:
            print(f"  [warn] No predictions found for disease_id={disease_id}")
        else:
            print(f"\nTop predictions (from {SPLIT} split, seed={SEED}, model={MODEL}):")
            print(preds.to_string(index=False))

        out_csv = PREDICTIONS_DIR / f"case_study_{case_label}_{MODEL}.csv"
        preds.to_csv(out_csv, index=False)
        print(f"  -> saved: {out_csv}")

        # KG path extraction for known approved drugs (as sanity check)
        known_drugs = kg[
            (kg["y_id"] == disease_id) &
            (kg["relation"].isin(["indication", "contraindication"]))
        ]["x_id"].unique()[:3]

        paths_out = []
        for drug_id in known_drugs:
            paths = extract_kg_paths(str(drug_id), str(disease_id), kg, max_paths=3)
            paths_out.extend(paths)

        if paths_out:
            paths_df = pd.DataFrame(paths_out)
            paths_file = PREDICTIONS_DIR / f"case_study_{case_label}_paths_{MODEL}.csv"
            paths_df.to_csv(paths_file, index=False)
            print(f"  KG paths -> {paths_file}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    print("Loading kg.csv for path extraction ...")
    kg = pd.read_csv(ROOT / "data" / "raw" / "kg.csv", low_memory=False)
    run_case_studies(kg)
