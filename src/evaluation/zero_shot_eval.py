"""
Zero-shot evaluation protocol.
For each held-out disease, score all candidate drugs, then compute per-disease AUPRC.
Also produces the degradation curve data for Q3.
"""

import json
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from src.evaluation.metrics import auprc, auroc


THERAPEUTIC_RELATIONS = {"indication", "contraindication"}


def evaluate_zero_shot(
    model: torch.nn.Module,
    h: dict,                        # pre-computed embeddings {node_type: Tensor}
    test_edges: pd.DataFrame,       # edges from zeroshot/seed_N/test.csv
    entity_index: dict,
    device: torch.device,
    drug_node_type: str = "drug",
    disease_node_type: str = "disease",
    top_k: int = 20,               # save top-K drug predictions per disease (for Q4)
) -> dict:
    """
    Per-disease zero-shot evaluation.
    For each test disease, score all drugs and compute AUPRC.

    Returns dict with:
      - per_disease: list of {disease_id, relation, n_pos, auprc, auroc}
      - aggregate: {relation: {auprc_mean, auprc_std, auroc_mean, auroc_std}}
      - degradation_curve: list of {disease_id, n_train_edges, zero_shot_auprc}
    """
    drug_embs = h[drug_node_type]          # [N_drugs, D]
    disease_embs = h[disease_node_type]    # [N_diseases, D]
    drug_idx_map = entity_index[drug_node_type]
    disease_idx_map = entity_index[disease_node_type]
    n_drugs = drug_embs.size(0)
    # Reverse lookup: drug_local_idx -> drug_id
    idx_to_drug_id = {v: k for k, v in drug_idx_map.items()}

    results_per_disease = []

    for (disease_id, relation), group in test_edges.groupby(["y_id", "relation"]):
        if disease_id not in disease_idx_map:
            continue
        dis_idx = disease_idx_map[disease_id]

        pos_drug_ids = set(group["x_id"].values)
        pos_drug_indices = [drug_idx_map[d] for d in pos_drug_ids if d in drug_idx_map]
        if not pos_drug_indices:
            continue

        # Score all drugs against this disease
        dis_emb = disease_embs[dis_idx].unsqueeze(0)  # [1, D]
        with torch.no_grad():
            all_scores = (drug_embs * dis_emb).sum(-1).cpu().numpy()  # [N_drugs]

        y_true = np.zeros(n_drugs)
        for pidx in pos_drug_indices:
            y_true[pidx] = 1.0

        if y_true.sum() == 0 or y_true.sum() == n_drugs:
            continue

        d_auprc = auprc(y_true, all_scores)
        d_auroc = auroc(y_true, all_scores)

        # Top-K drug predictions (for Q4 case studies)
        top_k_actual = min(top_k, n_drugs)
        topk_indices = np.argsort(all_scores)[::-1][:top_k_actual]
        top_drugs = [
            {
                "rank": int(i + 1),
                "drug_id": str(idx_to_drug_id.get(int(idx), f"idx_{idx}")),
                "score": float(round(all_scores[idx], 6)),
                "is_positive": bool(y_true[idx] > 0),
            }
            for i, idx in enumerate(topk_indices)
        ]

        results_per_disease.append({
            "disease_id": str(disease_id),
            "relation": relation,
            "n_pos": int(y_true.sum()),
            "auprc": round(d_auprc, 5),
            "auroc": round(d_auroc, 5),
            "top_k_drugs": top_drugs,
        })

    # Aggregate per relation
    aggregate = {}
    for rel in THERAPEUTIC_RELATIONS:
        rel_results = [r for r in results_per_disease if r["relation"] == rel]
        if not rel_results:
            aggregate[rel] = {"auprc_mean": None, "auprc_std": None,
                               "auroc_mean": None, "auroc_std": None, "n_diseases": 0}
            continue
        auprcs = [r["auprc"] for r in rel_results]
        aurocs = [r["auroc"] for r in rel_results]
        aggregate[rel] = {
            "auprc_mean": round(float(np.mean(auprcs)), 5),
            "auprc_std": round(float(np.std(auprcs)), 5),
            "auroc_mean": round(float(np.mean(aurocs)), 5),
            "auroc_std": round(float(np.std(aurocs)), 5),
            "n_diseases": len(rel_results),
        }

    return {
        "per_disease": results_per_disease,
        "aggregate": aggregate,
    }


def compute_degradation_curve_data(
    per_disease_results: list,
    train_edges: pd.DataFrame,
) -> list:
    """
    For Q3: maps each disease to (n_train_treatment_edges, zero_shot_auprc).
    Since these are zero-shot test diseases, all should have n_train_edges=0.
    This is most useful when comparing across model types:
      - Run the zero-shot model: n_train_edges=0 for all test diseases
      - Run a standard-split model: n_train_edges varies
    The curve shows how AUPRC degrades as n_train_edges approaches 0.
    """
    train_edge_counts = (
        train_edges[train_edges["relation"].isin(THERAPEUTIC_RELATIONS)]
        .groupby("y_id")
        .size()
        .to_dict()
    )
    curve = []
    for row in per_disease_results:
        did = row["disease_id"]
        n_train = train_edge_counts.get(did, 0)
        curve.append({
            "disease_id": did,
            "relation": row["relation"],
            "n_train_treatment_edges": int(n_train),
            "auprc": row["auprc"],
        })
    return curve
