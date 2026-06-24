"""
Phase 6: Reads all ablation result files and builds results/ablations/matrix.json.

Ablation matrix rows = variants, columns = (split, metric).
Q6 decision: compares txgnn vs txgnn_no_attn on zero-shot AUPRC.
             Sets "attention_optional" to True/False based on the 0.02 threshold.

All input from files. No numbers typed manually.
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = ROOT / "results" / "ablations"
OUTPUT = ROOT / "results" / "ablations" / "matrix.json"

VARIANTS = ["txgnn", "txgnn_no_attn", "txgnn_no_sim", "txgnn_no_both"]
SPLITS = ["standard", "zeroshot"]
SEEDS = [42, 0, 1]
TASKS = ["indication", "contraindication"]

# Q6 decision threshold (set before running, from CLAUDE.md)
ATTENTION_DELTA_THRESHOLD = 0.02


def load_result(variant: str, split: str, seed: int) -> dict | None:
    path = ABLATIONS_DIR / variant / split / f"seed_{seed}" / f"{variant}.json"
    if not path.exists() and variant == "txgnn":
        # Fall back to main TxGNN run (saves re-running the full model)
        path = ROOT / "results" / "txgnn" / split / f"seed_{seed}" / "txgnn.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_seeds(variant: str, split: str) -> dict:
    seed_results = [load_result(variant, split, s) for s in SEEDS]
    available = [r for r in seed_results if r is not None]

    row = {
        "variant": variant,
        "split": split,
        "n_seeds": len(available),
    }

    if not available:
        for task in TASKS:
            row[f"{task}_auprc_mean"] = "[NOT YET RUN]"
            row[f"{task}_auprc_std"] = "[NOT YET RUN]"
            row[f"{task}_auroc_mean"] = "[NOT YET RUN]"
        row["wall_clock_s_mean"] = "[NOT YET RUN]"
        return row

    for task in TASKS:
        # Prefer flat random-negative AUPRC for fair cross-model comparison
        flat_key = f"{task}_flat"
        def _get_auprc(r):
            if flat_key in r and r[flat_key].get("auprc") is not None:
                return r[flat_key]["auprc"]
            return r.get(task, {}).get("auprc")
        def _get_auroc(r):
            if flat_key in r and r[flat_key].get("auroc") is not None:
                return r[flat_key]["auroc"]
            return r.get(task, {}).get("auroc")
        auprcs = [_get_auprc(r) for r in available]
        auprcs = [v for v in auprcs if v is not None]
        aurocs = [_get_auroc(r) for r in available]
        aurocs = [v for v in aurocs if v is not None]
        row[f"{task}_auprc_mean"] = round(float(np.mean(auprcs)), 5) if auprcs else None
        row[f"{task}_auprc_std"] = round(float(np.std(auprcs)), 5) if len(auprcs) > 1 else None
        row[f"{task}_auroc_mean"] = round(float(np.mean(aurocs)), 5) if aurocs else None

    clocks = [r.get("wall_clock_seconds") for r in available if r.get("wall_clock_seconds")]
    row["wall_clock_s_mean"] = round(float(np.mean(clocks)), 1) if clocks else None

    return row


def compute_q6_decision(matrix_rows: list) -> dict:
    """
    Computes Q6 decision: is attention optional?
    Compares txgnn vs txgnn_no_attn on zero-shot split indication AUPRC.
    Decision rule from CLAUDE.md: delta < 0.02 -> optional; >= 0.02 -> matters.
    """
    full = next((r for r in matrix_rows
                 if r["variant"] == "txgnn" and r["split"] == "zeroshot"), None)
    no_attn = next((r for r in matrix_rows
                    if r["variant"] == "txgnn_no_attn" and r["split"] == "zeroshot"), None)

    if full is None or no_attn is None:
        return {"status": "NOT_YET_RUN", "detail": "One or both variants not yet run"}

    full_auprc = full.get("indication_auprc_mean")
    no_attn_auprc = no_attn.get("indication_auprc_mean")

    if full_auprc is None or no_attn_auprc is None or \
            full_auprc == "[NOT YET RUN]" or no_attn_auprc == "[NOT YET RUN]":
        return {"status": "NOT_YET_RUN", "detail": "AUPRC not yet available"}

    delta = round(float(full_auprc) - float(no_attn_auprc), 5)

    # Decision rule from CLAUDE.md (set before running):
    # delta < 0.02  -> attention not load-bearing (optional or harmful)
    # delta >= 0.02 -> attention is beneficial; report that
    # If delta is strongly negative (< -0.02), attention is actively detrimental.
    if delta >= ATTENTION_DELTA_THRESHOLD:
        verdict = "BENEFICIAL — attention improves zero-shot AUPRC"
    elif delta >= -ATTENTION_DELTA_THRESHOLD:
        verdict = "OPTIONAL — removing attention has minimal effect"
    else:
        verdict = "DETRIMENTAL — removing attention IMPROVES zero-shot AUPRC"

    return {
        "status": "DECIDED",
        "txgnn_attn_on_zeroshot_ind_auprc": full_auprc,
        "txgnn_attn_off_zeroshot_ind_auprc": no_attn_auprc,
        "delta_attn_on_minus_off": delta,
        "threshold": ATTENTION_DELTA_THRESHOLD,
        "attention_optional": abs(delta) < ATTENTION_DELTA_THRESHOLD,
        "conclusion": f"Attention is {verdict}. Delta = {delta:.5f}.",
    }


def build_matrix():
    rows = []
    for variant in VARIANTS:
        for split in SPLITS:
            rows.append(aggregate_seeds(variant, split))

    q6 = compute_q6_decision(rows)

    matrix = {
        "generated_from": "scripts/build_ablation_matrix.py",
        "source_dir": str(ABLATIONS_DIR),
        "seeds": SEEDS,
        "q6_decision": q6,
        "matrix": rows,
    }

    ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(matrix, f, indent=2)

    print(f"[matrix] Written to {OUTPUT}")
    print(f"\nQ6 Decision: {q6}")
    print("\nMatrix summary:")
    for row in rows:
        ind_ap = row.get("indication_auprc_mean", "?")
        con_ap = row.get("contraindication_auprc_mean", "?")
        print(f"  {row['variant']:20s} | {row['split']:10s} | "
              f"ind_auprc={ind_ap} | contra_auprc={con_ap}")

    return matrix


if __name__ == "__main__":
    build_matrix()
