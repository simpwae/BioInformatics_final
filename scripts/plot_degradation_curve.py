"""
Q3 — Degradation curve: AUPRC vs number of training treatment edges per disease.

Reads per-disease results from zero-shot eval (already in result JSON files).
Reads training edge counts from the training split file.
Writes:
  results/metrics/degradation_curve_data.json
  results/figures/degradation_curve.png

Only run after both GNN baseline and TxGNN zero-shot results exist.
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

# Models to compare (must have per_disease_results in their result JSONs)
# Each entry: (result_dir, result_filename, split_used_for_edge_counts)
MODELS = {
    "gnn_standard":   (ROOT / "results" / "gnn",   "gnn_baseline", "standard"),
    "gnn_zeroshot":   (ROOT / "results" / "gnn",   "gnn_baseline", "zeroshot"),
    "txgnn_zeroshot": (ROOT / "results" / "txgnn", "txgnn",        "zeroshot"),
}
SEED = 42


def load_per_disease(results_dir: Path, result_file: str, split: str) -> list:
    path = results_dir / split / f"seed_{SEED}" / f"{result_file}.json"
    if not path.exists():
        print(f"[warn] {path} not found — skipping")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("per_disease_results", [])


def load_train_edge_counts(split: str, seed: int = SEED) -> dict:
    """Returns {disease_id: n_train_treatment_edges}."""
    train_path = ROOT / "data" / "splits" / split / f"seed_{seed}" / "train.csv"
    if not train_path.exists():
        return {}
    train = pd.read_csv(train_path)
    th = train[train["relation"].isin(["indication", "contraindication"])]
    counts = th.groupby("y_id").size().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def build_curve_data():
    all_records = []
    for model_label, (results_dir, result_file, split) in MODELS.items():
        per_disease = load_per_disease(results_dir, result_file, split)
        train_counts = load_train_edge_counts(split)
        for row in per_disease:
            disease_id = str(row["disease_id"])
            n_train = train_counts.get(disease_id, 0)
            all_records.append({
                "model": model_label,
                "disease_id": disease_id,
                "relation": row["relation"],
                "n_train_edges": n_train,
                "auprc": row["auprc"],
            })

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "degradation_curve_data.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"[done] Curve data -> {out_path}")
    return all_records


def plot_curve(records: list):
    if not records:
        print("[skip] No records to plot.")
        return

    df = pd.DataFrame(records)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    relations = ["indication", "contraindication"]

    for ax, rel in zip(axes, relations):
        rel_df = df[df["relation"] == rel]
        if rel_df.empty:
            ax.set_title(f"{rel} (no data)")
            continue

        # Bin diseases by n_train_edges
        bins = [0, 1, 5, 10, 25, 50, 200]
        bin_labels = ["0", "1-4", "5-9", "10-24", "25-49", "50+"]
        rel_df = rel_df.copy()
        rel_df["bin"] = pd.cut(rel_df["n_train_edges"], bins=bins,
                               labels=bin_labels, right=False)

        for model_name in rel_df["model"].unique():
            m_df = rel_df[rel_df["model"] == model_name]
            grouped = m_df.groupby("bin", observed=True)["auprc"].agg(
                ["mean", "std", "count"]
            ).reset_index()
            ax.plot(
                grouped["bin"].astype(str),
                grouped["mean"],
                marker="o",
                label=f"{model_name} (n={grouped['count'].sum()})",
            )
            ax.fill_between(
                grouped["bin"].astype(str),
                grouped["mean"] - grouped["std"].fillna(0),
                grouped["mean"] + grouped["std"].fillna(0),
                alpha=0.15,
            )

        ax.set_title(f"{rel.capitalize()} — AUPRC vs Training Edges\n(Q3: seed={SEED}, standard=gnn; zeroshot=gnn+txgnn)")
        ax.set_xlabel("N training treatment edges per disease")
        ax.set_ylabel("AUPRC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "degradation_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[done] Figure -> {out_path}")
    plt.close()


if __name__ == "__main__":
    records = build_curve_data()
    plot_curve(records)
