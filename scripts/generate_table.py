"""
Generates the Q5 comparative table by reading from the results directory structure.

Actual result paths:
  results/gnn/{split}/seed_{n}/gnn_baseline.json      <- GNN + KG
  results/gnn/{split}/seed_{n}/gnn_no_kg.json          <- GNN no KG
  results/txgnn/{split}/seed_{n}/txgnn.json            <- TxGNN two-phase
  results/ablations/txgnn/{split}/seed_{n}/txgnn.json  <- TxGNN attn on
  results/ablations/txgnn_no_attn/{split}/seed_{n}/... <- TxGNN attn off
  results/alt_single_stage/{split}/seed_{n}/...        <- Q2 alternative A
  results/alt_joint_contrastive/{split}/seed_{n}/...   <- Q2 alternative B

Output:
  results/metrics/comparison_table.csv
  results/metrics/q2_compute_table.csv

NEVER edit these CSVs manually. Re-run this script to regenerate.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1]
SPLITS = ["standard", "zeroshot"]


MODEL_PATHS = {
    # Q1 / Q5 comparison
    "gnn_no_kg":        lambda split, seed: RESULTS / "gnn" / split / f"seed_{seed}" / "gnn_no_kg.json",
    "gnn_kg":           lambda split, seed: RESULTS / "gnn" / split / f"seed_{seed}" / "gnn_baseline.json",
    "txgnn_two_phase":  lambda split, seed: RESULTS / "txgnn" / split / f"seed_{seed}" / "txgnn.json",
    # Q2 alternatives
    "single_stage":     lambda split, seed: RESULTS / "alt_single_stage" / split / f"seed_{seed}" / "single_stage.json",
    "joint_contrastive":lambda split, seed: RESULTS / "alt_joint_contrastive" / split / f"seed_{seed}" / "joint_contrastive.json",
    # Q6 ablations
    "txgnn_attn_on":    lambda split, seed: RESULTS / "ablations" / "txgnn" / split / f"seed_{seed}" / "txgnn.json",
    "txgnn_attn_off":   lambda split, seed: RESULTS / "ablations" / "txgnn_no_attn" / split / f"seed_{seed}" / "txgnn_no_attn.json",
    # Q1 transformer pair
    "transformer_kg":   lambda split, seed: RESULTS / "transformer_kg" / split / f"seed_{seed}" / "transformer_kg.json",
    "transformer_nokg": lambda split, seed: RESULTS / "transformer_nokg" / split / f"seed_{seed}" / "transformer_nokg.json",
}


def load(path: Path, fallback: Path = None):
    if not path.exists() and fallback and fallback.exists():
        path = fallback
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_metric(result: dict, field: str, sub: str):
    """
    Tries 'indication_flat' first (TxGNN flat random-negative protocol),
    falls back to 'indication' (GNN baseline protocol).
    Both use the same random-negative 1:5 evaluation.
    """
    flat_key = f"{field}_flat"
    if flat_key in result and result[flat_key].get(sub) is not None:
        return result[flat_key][sub]
    return result.get(field, {}).get(sub)


def build_comparison_table():
    rows = []
    for model_name, path_fn in MODEL_PATHS.items():
        for split in SPLITS:
            # txgnn_attn_on falls back to main txgnn results if ablation re-run hasn't happened
            fallback_fn = None
            if model_name == "txgnn_attn_on":
                fallback_fn = lambda s, sp=split: RESULTS / "txgnn" / sp / f"seed_{s}" / "txgnn.json"
            results = [load(path_fn(split, s),
                           fallback_fn(s) if fallback_fn else None)
                       for s in SEEDS]
            available = [r for r in results if r is not None]

            base = {
                "model": model_name,
                "split": split,
                "n_seeds_run": len(available),
                "reproduction_type": available[0].get("reproduction_type", "?") if available else "?",
            }

            if not available:
                base.update({
                    "auprc_ind":       "[NOT YET RUN]",
                    "auprc_ind_std":   "",
                    "auroc_ind":       "[NOT YET RUN]",
                    "auroc_ind_std":   "",
                    "auprc_contra":    "[NOT YET RUN]",
                    "auprc_contra_std":"",
                    "auroc_contra":    "[NOT YET RUN]",
                    "auroc_contra_std":"",
                    "wall_s":          "[NOT YET RUN]",
                })
                rows.append(base)
                continue

            def _ms(vals):
                vals = [v for v in vals if v is not None]
                if not vals:
                    return None, None
                a = np.array(vals, dtype=float)
                return float(np.mean(a)), float(np.std(a))

            ind_auprc  = [_get_metric(r, "indication", "auprc")  for r in available]
            ind_auroc  = [_get_metric(r, "indication", "auroc")  for r in available]
            con_auprc  = [_get_metric(r, "contraindication", "auprc") for r in available]
            con_auroc  = [_get_metric(r, "contraindication", "auroc") for r in available]
            clocks     = [r.get("wall_clock_seconds") for r in available if r.get("wall_clock_seconds")]

            m_ia, s_ia = _ms(ind_auprc)
            m_iau, s_iau = _ms(ind_auroc)
            m_ca, s_ca = _ms(con_auprc)
            m_cau, s_cau = _ms(con_auroc)

            def _fmt(v):
                return round(v, 4) if v is not None else "[NOT YET RUN]"

            base.update({
                "auprc_ind":       _fmt(m_ia),
                "auprc_ind_std":   _fmt(s_ia),
                "auroc_ind":       _fmt(m_iau),
                "auroc_ind_std":   _fmt(s_iau),
                "auprc_contra":    _fmt(m_ca),
                "auprc_contra_std":_fmt(s_ca),
                "auroc_contra":    _fmt(m_cau),
                "auroc_contra_std":_fmt(s_cau),
                "wall_s":          round(np.mean(clocks), 1) if clocks else "[NOT YET RUN]",
            })
            rows.append(base)

    df = pd.DataFrame(rows)
    out = METRICS / "comparison_table.csv"
    df.to_csv(out, index=False)
    print(f"[table] Written to {out}")
    print(df.to_string(index=False))
    return df


def build_q6_ablation_table():
    """Q6: attention on vs. off on zero-shot split."""
    rows = []
    for variant in ["txgnn_attn_on", "txgnn_attn_off"]:
        path_fn = MODEL_PATHS[variant]
        for split in SPLITS:
            fallback_fn = None
            if variant == "txgnn_attn_on":
                fallback_fn = lambda s, sp=split: RESULTS / "txgnn" / sp / f"seed_{s}" / "txgnn.json"
            results = [load(path_fn(split, s),
                           fallback_fn(s) if fallback_fn else None)
                       for s in SEEDS]
            available = [r for r in results if r is not None]
            if not available:
                rows.append({"variant": variant, "split": split, "auprc_ind_mean": "[NOT YET RUN]"})
                continue
            vals = [_get_metric(r, "indication", "auprc") for r in available]
            vals = [v for v in vals if v is not None]
            rows.append({
                "variant": variant,
                "split": split,
                "auprc_ind_mean": round(float(np.mean(vals)), 4) if vals else None,
                "auprc_ind_std": round(float(np.std(vals)), 4) if vals else None,
                "n_seeds": len(vals),
            })

    df = pd.DataFrame(rows)
    if len(df) == 4:  # 2 variants × 2 splits
        zs_on = df[(df.variant == "txgnn_attn_on") & (df.split == "zeroshot")]["auprc_ind_mean"].values
        zs_off = df[(df.variant == "txgnn_attn_off") & (df.split == "zeroshot")]["auprc_ind_mean"].values
        if len(zs_on) and len(zs_off) and isinstance(zs_on[0], float) and isinstance(zs_off[0], float):
            delta = zs_on[0] - zs_off[0]
            verdict = "attention optional (delta < 0.02)" if delta < 0.02 else "attention matters (delta >= 0.02)"
            print(f"\n[Q6] Zero-shot delta = {delta:.4f} — {verdict}")

    out = METRICS / "q6_ablation_table.csv"
    df.to_csv(out, index=False)
    print(f"[q6] Written to {out}")
    return df


if __name__ == "__main__":
    print("=== Q5 Comparison Table ===")
    build_comparison_table()
    print("\n=== Q6 Ablation Table ===")
    build_q6_ablation_table()
