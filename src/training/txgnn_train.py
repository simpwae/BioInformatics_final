"""
TxGNN two-phase fine-tuning (Phase 2 of reproduction).
Includes disease-similarity metric-learning loss.

Used by: TxGNN, TxGNN-no-attn, TxGNN-no-sim (ablations).
Also used as Phase 2 for SingleStage (minus the Phase 1 separation).
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.amp import GradScaler, autocast
import pandas as pd

from src.evaluation.metrics import evaluate_flat, auprc as _auprc, auroc as _auroc
from src.evaluation.zero_shot_eval import evaluate_zero_shot

THERAPEUTIC_RELATIONS = {"indication", "contraindication"}


def _build_disease_positive_pairs(train_edges: pd.DataFrame, entity_index: dict,
                                  disease_type: str = "disease") -> torch.Tensor:
    """
    Returns pairs of disease indices that share at least one approved drug.
    Used for the metric-learning loss.
    """
    disease_id_map = entity_index.get(disease_type, {})
    drug_to_diseases: dict = {}

    for _, row in train_edges[train_edges["relation"].isin(THERAPEUTIC_RELATIONS)].iterrows():
        drug_id, disease_id = row["x_id"], row["y_id"]
        if disease_id not in disease_id_map:
            continue
        drug_to_diseases.setdefault(drug_id, set()).add(disease_id_map[disease_id])

    pairs = []
    for _, diseases in drug_to_diseases.items():
        disease_list = list(diseases)
        for i in range(len(disease_list)):
            for j in range(i + 1, len(disease_list)):
                pairs.append((disease_list[i], disease_list[j]))
                if len(pairs) >= 5000:
                    break
            if len(pairs) >= 5000:
                break

    if not pairs:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


def finetune_txgnn(
    model,
    data,
    entity_index: dict,
    split_dir: Path,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    neg_ratio: int = 5,
    mixed_precision: bool = True,
    results_dir: Path = None,
    model_name: str = "txgnn",
    split_name: str = "standard",
    seed: int = 42,
    patience: int = 10,
    sim_loss_weight: float = 0.3,
    drug_node_type: str = "drug",
    disease_node_type: str = "disease",
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    n_diseases = data[disease_node_type].num_nodes
    n_drugs = data[drug_node_type].num_nodes

    train_edges = pd.read_csv(split_dir / "train.csv")
    val_edges = pd.read_csv(split_dir / "val.csv")
    test_edges = pd.read_csv(split_dir / "test.csv")

    # Disease positive pairs for metric learning loss
    pos_pairs = _build_disease_positive_pairs(
        train_edges, entity_index, disease_node_type
    ).to(device)

    disease_id_map = entity_index.get(disease_node_type, {})
    drug_id_map = entity_index.get(drug_node_type, {})

    def _build_edge_tensors(df):
        pairs = {}
        for rel, grp in df.groupby("relation"):
            valid = [(drug_id_map[r["x_id"]], disease_id_map[r["y_id"]])
                     for _, r in grp.iterrows()
                     if r["x_id"] in drug_id_map and r["y_id"] in disease_id_map]
            if not valid:
                continue
            pairs[rel] = (
                torch.tensor([v[0] for v in valid], dtype=torch.long),
                torch.tensor([v[1] for v in valid], dtype=torch.long),
            )
        return pairs

    train_pairs = _build_edge_tensors(train_edges)
    val_pairs = _build_edge_tensors(val_edges)

    # Seen disease embeddings (for zero-shot transfer at test time)
    seen_disease_ids = set(train_edges[
        train_edges["relation"].isin(THERAPEUTIC_RELATIONS)
    ]["y_id"].unique())
    seen_disease_indices = torch.tensor(
        [disease_id_map[d] for d in seen_disease_ids if d in disease_id_map],
        dtype=torch.long, device=device,
    )

    best_val_auprc = -1.0
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        with autocast("cuda", enabled=mixed_precision):
            h = model.encode(edge_index_dict)

            task_loss = torch.tensor(0.0, device=device)
            n_tasks = 0

            for rel, (drug_idx, disease_idx) in train_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                pos_scores = (h[drug_node_type][drug_idx] *
                              h[disease_node_type][disease_idx]).sum(-1)
                neg_dis = torch.randint(0, n_diseases,
                                        (len(drug_idx) * neg_ratio,), device=device)
                neg_drug = drug_idx.repeat(neg_ratio)
                neg_scores = (h[drug_node_type][neg_drug] *
                              h[disease_node_type][neg_dis]).sum(-1)
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([torch.ones_like(pos_scores),
                                    torch.zeros_like(neg_scores)])
                task_loss = task_loss + F.binary_cross_entropy_with_logits(scores, labels)
                n_tasks += 1

            if n_tasks > 0:
                task_loss = task_loss / n_tasks

            # Disease similarity metric-learning loss
            sim_loss = torch.tensor(0.0, device=device)
            if (model.similarity_module is not None and
                    pos_pairs.size(0) > 0 and sim_loss_weight > 0):
                disease_embs = h[disease_node_type]  # [N_diseases, D]
                # Random negative pairs
                neg_i = torch.randint(0, n_diseases, (min(pos_pairs.size(0), 500),), device=device)
                neg_j = torch.randint(0, n_diseases, (min(pos_pairs.size(0), 500),), device=device)
                neg_pairs = torch.stack([neg_i, neg_j], dim=1)
                # Use a random subset of positive pairs
                perm = torch.randperm(pos_pairs.size(0), device=device)[:500]
                sim_loss = model.similarity_module.metric_learning_loss(
                    disease_embs, pos_pairs[perm], neg_pairs
                )

            total_loss = task_loss + sim_loss_weight * sim_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                with autocast("cuda", enabled=mixed_precision):
                    h_val = model.encode(edge_index_dict)

            val_auprcs = []
            for rel, (drug_idx, disease_idx) in val_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                pos_s = (h_val[drug_node_type][drug_idx] *
                         h_val[disease_node_type][disease_idx]).sum(-1).cpu().numpy()
                neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * 5,), device=device)
                neg_drug = drug_idx.repeat(5)
                neg_s = (h_val[drug_node_type][neg_drug] *
                         h_val[disease_node_type][neg_dis]).sum(-1).cpu().numpy()
                y_t = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
                y_s = np.concatenate([pos_s, neg_s])
                from src.evaluation.metrics import auprc as _auprc
                val_auprcs.append(_auprc(y_t, y_s))

            val_auprc = float(np.mean(val_auprcs)) if val_auprcs else 0.0
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5

            if no_improve >= patience and epoch > 20:
                print(f"  [txgnn] Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        with autocast("cuda", enabled=mixed_precision):
            h_test = model.encode(edge_index_dict)
    h_test = {nt: emb.detach() for nt, emb in h_test.items()}

    support_embs = h_test[disease_node_type][seen_disease_indices]

    # Flat random-negative evaluation (same protocol as GNN baseline finetune.py)
    # This enables apples-to-apples Q5 comparison across all models.
    test_pairs_flat = {}
    for rel, grp in test_edges.groupby("relation"):
        valid = [(drug_id_map[r["x_id"]], disease_id_map[r["y_id"]])
                 for _, r in grp.iterrows()
                 if r["x_id"] in drug_id_map and r["y_id"] in disease_id_map]
        if not valid:
            continue
        test_pairs_flat[rel] = (
            torch.tensor([v[0] for v in valid], dtype=torch.long),
            torch.tensor([v[1] for v in valid], dtype=torch.long),
        )

    flat_metrics = {}
    for rel, (d_idx, dis_idx) in test_pairs_flat.items():
        d_idx, dis_idx = d_idx.to(device), dis_idx.to(device)
        ps = (h_test[drug_node_type][d_idx] * h_test[disease_node_type][dis_idx]).sum(-1).cpu().numpy()
        nd = torch.randint(0, n_diseases, (len(d_idx) * 5,), device=device)
        nd_drug = d_idx.repeat(5)
        ns = (h_test[drug_node_type][nd_drug] * h_test[disease_node_type][nd]).sum(-1).cpu().numpy()
        yt = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
        ys = np.concatenate([ps, ns])
        flat_metrics[rel] = {"auprc": _auprc(yt, ys), "auroc": _auroc(yt, ys), "n_pairs": len(yt)}

    # Zero-shot eval (per-disease, full drug ranking)
    zs_results = evaluate_zero_shot(
        model=model,
        h=h_test,
        test_edges=test_edges,
        entity_index=entity_index,
        device=device,
        drug_node_type=drug_node_type,
        disease_node_type=disease_node_type,
    )

    ind_agg = zs_results["aggregate"].get("indication", {})
    contra_agg = zs_results["aggregate"].get("contraindication", {})

    model_cfg = model.model_config() if hasattr(model, "model_config") else {}

    results = {
        "model": model_name,
        "reproduction_type": model_cfg.get("reproduction_type", "scaled_reproduction"),
        "split": split_name,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                   if torch.cuda.is_available() else 0,
        "date": __import__("datetime").date.today().isoformat(),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "model_config": model_cfg,
        "indication": {
            "auprc": ind_agg.get("auprc_mean"),
            "auprc_std": ind_agg.get("auprc_std"),
            "auroc": ind_agg.get("auroc_mean"),
            "n_test_diseases": ind_agg.get("n_diseases", 0),
        },
        "contraindication": {
            "auprc": contra_agg.get("auprc_mean"),
            "auprc_std": contra_agg.get("auprc_std"),
            "auroc": contra_agg.get("auroc_mean"),
            "n_test_diseases": contra_agg.get("n_diseases", 0),
        },
        "per_disease_results": zs_results["per_disease"],
        # Flat random-negative metrics for cross-model comparison (same as GNN baseline protocol)
        "indication_flat": {
            "auprc": flat_metrics.get("indication", {}).get("auprc"),
            "auroc": flat_metrics.get("indication", {}).get("auroc"),
            "n_test_pairs": flat_metrics.get("indication", {}).get("n_pairs"),
        },
        "contraindication_flat": {
            "auprc": flat_metrics.get("contraindication", {}).get("auprc"),
            "auroc": flat_metrics.get("contraindication", {}).get("auroc"),
            "n_test_pairs": flat_metrics.get("contraindication", {}).get("n_pairs"),
        },
        "notes": f"best_val_auprc={best_val_auprc:.4f}",
    }

    if results_dir is not None:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [txgnn] results -> {out_path}")

    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind_auprc={results['indication']['auprc']} | "
          f"contra_auprc={results['contraindication']['auprc']} | "
          f"wall={results['wall_clock_seconds']}s")
    return results
