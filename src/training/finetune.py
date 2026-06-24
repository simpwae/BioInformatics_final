"""
Phase 2 training: fine-tune on therapeutic (indication / contraindication) edges.
For the GNN baseline: no metric-learning module — just binary classification.
For TxGNN: metric-learning module added on top (implemented in txgnn.py).

Writes results to results/gnn/{split}/seed_{n}.json following the schema in CLAUDE.md.
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.amp import GradScaler, autocast
import pandas as pd

from src.evaluation.metrics import evaluate_flat


THERAPEUTIC_RELATIONS = {"indication", "contraindication"}


def _load_split_edges(split_dir: Path, subset: str) -> pd.DataFrame:
    path = split_dir / f"{subset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(path)


def _build_pairs(
    edges: pd.DataFrame,
    entity_index: dict,
    drug_type: str = "drug",
    disease_type: str = "disease",
) -> dict:
    """
    Extracts positive (drug_idx, disease_idx, label) pairs for one relation type.
    Returns dict: relation -> (src_idx, dst_idx).
    """
    result = {}
    for rel, group in edges.groupby("relation"):
        drug_ids = group["x_id"].values
        disease_ids = group["y_id"].values
        valid = [
            (drug_ids[i], disease_ids[i])
            for i in range(len(drug_ids))
            if drug_ids[i] in entity_index.get(drug_type, {})
            and disease_ids[i] in entity_index.get(disease_type, {})
        ]
        if not valid:
            continue
        d_idx = torch.tensor([entity_index[drug_type][d] for d, _ in valid], dtype=torch.long)
        dis_idx = torch.tensor([entity_index[disease_type][d] for _, d in valid], dtype=torch.long)
        result[rel] = (d_idx, dis_idx)
    return result


def negative_sample_diseases(
    drug_idx: torch.Tensor,
    pos_disease_idx: torch.Tensor,
    n_diseases: int,
    device: torch.device,
    neg_ratio: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt the disease end of each positive pair."""
    neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * neg_ratio,), device=device)
    neg_drug = drug_idx.repeat(neg_ratio)
    return neg_drug, neg_dis


def finetune(
    model: torch.nn.Module,
    data,               # HeteroData on device
    entity_index: dict,
    split_dir: Path,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    neg_ratio: int = 5,
    mixed_precision: bool = True,
    results_dir: Path = None,
    model_name: str = "gnn_baseline",
    split_name: str = "standard",
    seed: int = 42,
    patience: int = 10,
    drug_node_type: str = "drug",
    disease_node_type: str = "disease",
) -> dict:
    """
    Fine-tunes model on therapeutic edges from the split.
    Returns the results dict (also written to disk).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    n_diseases = data[disease_node_type].num_nodes

    # Load split edges
    train_edges = _load_split_edges(split_dir, "train")
    val_edges = _load_split_edges(split_dir, "val")
    test_edges = _load_split_edges(split_dir, "test")

    train_pairs = _build_pairs(train_edges, entity_index, drug_node_type, disease_node_type)
    val_pairs = _build_pairs(val_edges, entity_index, drug_node_type, disease_node_type)
    test_pairs = _build_pairs(test_edges, entity_index, drug_node_type, disease_node_type)

    best_val_auprc = -1
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        with autocast("cuda", enabled=mixed_precision):
            h = model.encode(edge_index_dict)
            total_loss = torch.tensor(0.0, device=device)
            n_tasks = 0

            for rel, (drug_idx, disease_idx) in train_pairs.items():
                drug_idx = drug_idx.to(device)
                disease_idx = disease_idx.to(device)

                pos_drug_emb = h[drug_node_type][drug_idx]
                pos_dis_emb = h[disease_node_type][disease_idx]
                pos_scores = (pos_drug_emb * pos_dis_emb).sum(-1)

                neg_drug, neg_dis = negative_sample_diseases(
                    drug_idx, disease_idx, n_diseases, device, neg_ratio)
                neg_drug_emb = h[drug_node_type][neg_drug]
                neg_dis_emb = h[disease_node_type][neg_dis]
                neg_scores = (neg_drug_emb * neg_dis_emb).sum(-1)

                pos_labels = torch.ones_like(pos_scores)
                neg_labels = torch.zeros_like(neg_scores)
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([pos_labels, neg_labels])
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                total_loss = total_loss + loss
                n_tasks += 1

            if n_tasks > 0:
                total_loss = total_loss / n_tasks

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Validation check every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = _evaluate_split(model, h.detach() if hasattr(h, 'detach') else
                                          _recompute_h(model, edge_index_dict),
                                          val_pairs, device, drug_node_type, disease_node_type)
            val_auprc = np.mean([v["auprc"] for v in val_metrics.values() if "auprc" in v])
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5

            if no_improve >= patience:
                print(f"  [finetune] Early stop at epoch {epoch}")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        with autocast("cuda", enabled=mixed_precision):
            h = model.encode(edge_index_dict)
    h = {nt: emb.detach() for nt, emb in h.items()}

    test_metrics = _evaluate_split(model, h, test_pairs, device, drug_node_type, disease_node_type)
    wall_s = round(time.time() - t0, 1)

    # Build results dict following CLAUDE.md schema
    ind = test_metrics.get("indication", {})
    contra = test_metrics.get("contraindication", {})

    results = {
        "model": model_name,
        "reproduction_type": "scaled_reproduction",
        "split": split_name,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                   if torch.cuda.is_available() else 0,
        "date": __import__("datetime").date.today().isoformat(),
        "wall_clock_seconds": wall_s,
        "indication": {
            "auprc": ind.get("auprc", None),
            "auroc": ind.get("auroc", None),
            "n_test_pairs": ind.get("n_pairs", 0),
        },
        "contraindication": {
            "auprc": contra.get("auprc", None),
            "auroc": contra.get("auroc", None),
            "n_test_pairs": contra.get("n_pairs", 0),
        },
        "notes": f"best_val_auprc={best_val_auprc:.4f}",
    }

    if results_dir is not None:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [finetune] results -> {out_path}")

    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind_auprc={results['indication']['auprc']} | "
          f"contra_auprc={results['contraindication']['auprc']} | "
          f"wall={wall_s}s")
    return results


def _recompute_h(model, edge_index_dict):
    model.eval()
    with torch.no_grad():
        return model.encode(edge_index_dict)


def _evaluate_split(model, h, pairs, device, drug_type, disease_type) -> dict:
    """Returns dict: relation -> metrics dict."""
    results = {}
    for rel, (drug_idx, disease_idx) in pairs.items():
        drug_idx = drug_idx.to(device)
        disease_idx = disease_idx.to(device)

        n_drugs = h[drug_type].size(0)
        n_diseases = h[disease_type].size(0)

        # Build full drug × disease score matrix for evaluation
        # For large KGs this is done per-disease query
        with torch.no_grad():
            pos_drug_emb = h[drug_type][drug_idx]
            pos_dis_emb = h[disease_type][disease_idx]
            pos_scores = (pos_drug_emb * pos_dis_emb).sum(-1).cpu().numpy()

        n_pos = len(pos_scores)
        # Random negatives for flat evaluation
        neg_disease_idx = torch.randint(0, n_diseases, (n_pos * 5,), device=device)
        neg_drug_idx = drug_idx.repeat(5)
        with torch.no_grad():
            neg_drug_emb = h[drug_type][neg_drug_idx]
            neg_dis_emb = h[disease_type][neg_disease_idx]
            neg_scores = (neg_drug_emb * neg_dis_emb).sum(-1).cpu().numpy()

        y_true = np.concatenate([np.ones(n_pos), np.zeros(len(neg_scores))])
        y_score = np.concatenate([pos_scores, neg_scores])

        from src.evaluation.metrics import evaluate_flat
        results[rel] = evaluate_flat(y_true, y_score, rel)

    return results
