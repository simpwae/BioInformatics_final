"""
Phase 5 — Single-stage training loop.
No Phase 1/Phase 2 separation. KG link prediction + therapeutic task from epoch 1.

Loss = alpha * L_kg_lp + (1 - alpha) * L_therapeutic + beta * L_contrastive

Writes results using the same schema as txgnn_train.py.
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.amp import GradScaler, autocast
import pandas as pd

from src.evaluation.zero_shot_eval import evaluate_zero_shot
from src.evaluation.metrics import auprc as _auprc, auroc as _auroc

THERAPEUTIC_RELATIONS = {"indication", "contraindication"}


def _negative_sample(ei, n_dst, device, ratio=1):
    n = ei.size(1)
    neg_tail = torch.randint(0, n_dst, (n * ratio,), device=device)
    neg_head = ei[0].repeat(ratio)
    return torch.stack([neg_head, neg_tail], dim=0)


def train_single_stage(
    model,
    data,
    entity_index: dict,
    split_dir: Path,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    neg_ratio: int = 5,
    kg_loss_weight: float = 0.5,
    mixed_precision: bool = True,
    results_dir: Path = None,
    model_name: str = "single_stage",
    split_name: str = "standard",
    seed: int = 42,
    patience: int = 10,
    drug_node_type: str = "drug",
    disease_node_type: str = "disease",
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    n_diseases = data[disease_node_type].num_nodes

    train_edges = pd.read_csv(split_dir / "train.csv")
    val_edges = pd.read_csv(split_dir / "val.csv")
    test_edges = pd.read_csv(split_dir / "test.csv")

    drug_id_map = entity_index.get(drug_node_type, {})
    disease_id_map = entity_index.get(disease_node_type, {})

    def _edge_tensors(df):
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

    train_pairs = _edge_tensors(train_edges)
    val_pairs = _edge_tensors(val_edges)

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

            # KG link prediction loss (all edge types)
            kg_loss = torch.tensor(0.0, device=device)
            n_kg = 0
            for (src_t, rel, dst_t), ei in edge_index_dict.items():
                if ei.size(1) == 0:
                    continue
                pos_s = (h[src_t][ei[0]] * h[dst_t][ei[1]]).sum(-1)
                neg_ei = _negative_sample(ei, data[dst_t].num_nodes, device)
                neg_s = (h[src_t][neg_ei[0]] * h[dst_t][neg_ei[1]]).sum(-1)
                scores = torch.cat([pos_s, neg_s])
                labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
                kg_loss = kg_loss + F.binary_cross_entropy_with_logits(scores, labels)
                n_kg += 1
            if n_kg > 0:
                kg_loss = kg_loss / n_kg

            # Therapeutic task loss
            task_loss = torch.tensor(0.0, device=device)
            n_tasks = 0
            for rel, (drug_idx, disease_idx) in train_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                pos_s = (h[drug_node_type][drug_idx] *
                         h[disease_node_type][disease_idx]).sum(-1)
                neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * neg_ratio,), device=device)
                neg_drug = drug_idx.repeat(neg_ratio)
                neg_s = (h[drug_node_type][neg_drug] *
                         h[disease_node_type][neg_dis]).sum(-1)
                scores = torch.cat([pos_s, neg_s])
                labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
                task_loss = task_loss + F.binary_cross_entropy_with_logits(scores, labels)
                n_tasks += 1
            if n_tasks > 0:
                task_loss = task_loss / n_tasks

            total_loss = kg_loss_weight * kg_loss + (1 - kg_loss_weight) * task_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                with autocast("cuda", enabled=mixed_precision):
                    h_v = model.encode(edge_index_dict)
            val_auprcs = []
            for rel, (drug_idx, disease_idx) in val_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                ps = (h_v[drug_node_type][drug_idx] * h_v[disease_node_type][disease_idx]).sum(-1).cpu().numpy()
                nd = torch.randint(0, n_diseases, (len(drug_idx) * 5,), device=device)
                nd_drug = drug_idx.repeat(5)
                ns = (h_v[drug_node_type][nd_drug] * h_v[disease_node_type][nd]).sum(-1).cpu().numpy()
                yt = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
                ys = np.concatenate([ps, ns])
                val_auprcs.append(_auprc(yt, ys))
            vap = float(np.mean(val_auprcs)) if val_auprcs else 0.0
            if vap > best_val_auprc:
                best_val_auprc = vap
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience and epoch > 20:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        with autocast("cuda", enabled=mixed_precision):
            h_test = model.encode(edge_index_dict)
    h_test = {nt: emb.detach() for nt, emb in h_test.items()}

    zs = evaluate_zero_shot(model, h_test, test_edges, entity_index, device,
                             drug_node_type, disease_node_type)
    ind = zs["aggregate"].get("indication", {})
    contra = zs["aggregate"].get("contraindication", {})

    # Flat random-negative evaluation for cross-model comparison
    flat_metrics = {}
    for rel, grp in test_edges.groupby("relation"):
        if rel not in THERAPEUTIC_RELATIONS:
            continue
        valid = [(drug_id_map[r["x_id"]], disease_id_map[r["y_id"]])
                 for _, r in grp.iterrows()
                 if r["x_id"] in drug_id_map and r["y_id"] in disease_id_map]
        if not valid:
            continue
        d_idx = torch.tensor([v[0] for v in valid], dtype=torch.long, device=device)
        dis_idx = torch.tensor([v[1] for v in valid], dtype=torch.long, device=device)
        ps = (h_test[drug_node_type][d_idx] * h_test[disease_node_type][dis_idx]).sum(-1).cpu().numpy()
        nd = torch.randint(0, n_diseases, (len(d_idx) * 5,), device=device)
        nd_drug = d_idx.repeat(5)
        ns = (h_test[drug_node_type][nd_drug] * h_test[disease_node_type][nd]).sum(-1).cpu().numpy()
        yt = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
        ys = np.concatenate([ps, ns])
        flat_metrics[rel] = {"auprc": _auprc(yt, ys), "auroc": _auroc(yt, ys), "n_pairs": len(yt)}

    model_cfg = model.model_config() if hasattr(model, "model_config") else {}

    results = {
        "model": model_name,
        "reproduction_type": "original_ablation",
        "split": split_name,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                   if torch.cuda.is_available() else 0,
        "date": __import__("datetime").date.today().isoformat(),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "model_config": model_cfg,
        "indication": {
            "auprc": ind.get("auprc_mean"),
            "auroc": ind.get("auroc_mean"),
            "n_test_diseases": ind.get("n_diseases", 0),
        },
        "contraindication": {
            "auprc": contra.get("auprc_mean"),
            "auroc": contra.get("auroc_mean"),
            "n_test_diseases": contra.get("n_diseases", 0),
        },
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
        "compute": {
            "kg_loss_weight": kg_loss_weight,
            "n_epochs_run": epoch,
        },
        "notes": f"single_stage|best_val_auprc={best_val_auprc:.4f}",
    }

    if results_dir:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{model_name}.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind_auprc={results['indication']['auprc']} | wall={results['wall_clock_seconds']}s")
    return results


def train_joint_contrastive(
    model,
    data,
    entity_index: dict,
    split_dir: Path,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    neg_ratio: int = 5,
    contrastive_weight: float = 0.3,
    mixed_precision: bool = True,
    results_dir: Path = None,
    model_name: str = "joint_contrastive",
    split_name: str = "standard",
    seed: int = 42,
    patience: int = 10,
    drug_node_type: str = "drug",
    disease_node_type: str = "disease",
    pairs_per_epoch: int = 128,
) -> dict:
    """
    Alternative B training: therapeutic loss + InfoNCE disease-similarity loss.
    No KG link-prediction loss and no pretraining phase.
    """
    from collections import defaultdict

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    n_diseases = data[disease_node_type].num_nodes

    train_edges = pd.read_csv(split_dir / "train.csv")
    val_edges = pd.read_csv(split_dir / "val.csv")
    test_edges = pd.read_csv(split_dir / "test.csv")

    drug_id_map = entity_index.get(drug_node_type, {})
    disease_id_map = entity_index.get(disease_node_type, {})

    def _edge_tensors(df):
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

    train_pairs = _edge_tensors(train_edges)
    val_pairs = _edge_tensors(val_edges)

    # Precompute positive disease pairs (share >= 1 drug in train therapeutic edges)
    disease_to_drugs: dict = defaultdict(set)
    for rel in THERAPEUTIC_RELATIONS:
        for _, row in train_edges[train_edges["relation"] == rel].iterrows():
            d_id = row["y_id"]
            drug_id = row["x_id"]
            if d_id in disease_id_map and drug_id in drug_id_map:
                disease_to_drugs[disease_id_map[d_id]].add(drug_id_map[drug_id])

    positive_disease_pairs = []
    disease_list = list(disease_to_drugs.keys())
    for i in range(len(disease_list)):
        for j in range(i + 1, len(disease_list)):
            if disease_to_drugs[disease_list[i]] & disease_to_drugs[disease_list[j]]:
                positive_disease_pairs.append((disease_list[i], disease_list[j]))

    print(f"  [jc] Found {len(positive_disease_pairs)} positive disease pairs for InfoNCE")
    positive_disease_pairs_t = (
        torch.tensor(positive_disease_pairs, dtype=torch.long, device=device)
        if positive_disease_pairs else None
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

            # Therapeutic task loss
            task_loss = torch.tensor(0.0, device=device)
            n_tasks = 0
            for rel, (drug_idx, disease_idx) in train_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                pos_s = (h[drug_node_type][drug_idx] *
                         h[disease_node_type][disease_idx]).sum(-1)
                neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * neg_ratio,), device=device)
                neg_drug = drug_idx.repeat(neg_ratio)
                neg_s = (h[drug_node_type][neg_drug] *
                         h[disease_node_type][neg_dis]).sum(-1)
                scores = torch.cat([pos_s, neg_s])
                labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
                task_loss = task_loss + F.binary_cross_entropy_with_logits(scores, labels)
                n_tasks += 1
            if n_tasks > 0:
                task_loss = task_loss / n_tasks

            # InfoNCE contrastive disease-similarity loss
            contrastive_loss = torch.tensor(0.0, device=device)
            if positive_disease_pairs_t is not None and hasattr(model, "infonce_disease_loss"):
                n_pairs = min(pairs_per_epoch, positive_disease_pairs_t.size(0))
                idx = torch.randperm(positive_disease_pairs_t.size(0), device=device)[:n_pairs]
                sampled_pairs = positive_disease_pairs_t[idx]
                contrastive_loss = model.infonce_disease_loss(
                    h[disease_node_type], sampled_pairs
                )

            total_loss = task_loss + contrastive_weight * contrastive_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                with autocast("cuda", enabled=mixed_precision):
                    h_v = model.encode(edge_index_dict)
            val_auprcs = []
            for rel, (drug_idx, disease_idx) in val_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                ps = (h_v[drug_node_type][drug_idx] * h_v[disease_node_type][disease_idx]).sum(-1).cpu().numpy()
                nd = torch.randint(0, n_diseases, (len(drug_idx) * 5,), device=device)
                nd_drug = drug_idx.repeat(5)
                ns = (h_v[drug_node_type][nd_drug] * h_v[disease_node_type][nd]).sum(-1).cpu().numpy()
                yt = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
                ys = np.concatenate([ps, ns])
                val_auprcs.append(_auprc(yt, ys))
            vap = float(np.mean(val_auprcs)) if val_auprcs else 0.0
            if vap > best_val_auprc:
                best_val_auprc = vap
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience and epoch > 20:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        with autocast("cuda", enabled=mixed_precision):
            h_test = model.encode(edge_index_dict)
    h_test = {nt: emb.detach() for nt, emb in h_test.items()}

    zs = evaluate_zero_shot(model, h_test, test_edges, entity_index, device,
                             drug_node_type, disease_node_type)
    ind = zs["aggregate"].get("indication", {})
    contra = zs["aggregate"].get("contraindication", {})

    flat_metrics = {}
    for rel, grp in test_edges.groupby("relation"):
        if rel not in THERAPEUTIC_RELATIONS:
            continue
        valid = [(drug_id_map[r["x_id"]], disease_id_map[r["y_id"]])
                 for _, r in grp.iterrows()
                 if r["x_id"] in drug_id_map and r["y_id"] in disease_id_map]
        if not valid:
            continue
        d_idx = torch.tensor([v[0] for v in valid], dtype=torch.long, device=device)
        dis_idx = torch.tensor([v[1] for v in valid], dtype=torch.long, device=device)
        ps = (h_test[drug_node_type][d_idx] * h_test[disease_node_type][dis_idx]).sum(-1).cpu().numpy()
        nd = torch.randint(0, n_diseases, (len(d_idx) * 5,), device=device)
        nd_drug = d_idx.repeat(5)
        ns = (h_test[drug_node_type][nd_drug] * h_test[disease_node_type][nd]).sum(-1).cpu().numpy()
        yt = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
        ys = np.concatenate([ps, ns])
        flat_metrics[rel] = {"auprc": _auprc(yt, ys), "auroc": _auroc(yt, ys), "n_pairs": len(yt)}

    model_cfg = model.model_config() if hasattr(model, "model_config") else {}

    results = {
        "model": model_name,
        "reproduction_type": "original_ablation",
        "split": split_name,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                   if torch.cuda.is_available() else 0,
        "date": __import__("datetime").date.today().isoformat(),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "model_config": model_cfg,
        "n_positive_disease_pairs": len(positive_disease_pairs),
        "indication": {
            "auprc": ind.get("auprc_mean"),
            "auroc": ind.get("auroc_mean"),
            "n_test_diseases": ind.get("n_diseases", 0),
        },
        "contraindication": {
            "auprc": contra.get("auprc_mean"),
            "auroc": contra.get("auroc_mean"),
            "n_test_diseases": contra.get("n_diseases", 0),
        },
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
        "compute": {
            "contrastive_weight": contrastive_weight,
            "n_epochs_run": epoch,
        },
        "notes": f"joint_contrastive|best_val_auprc={best_val_auprc:.4f}",
    }

    if results_dir:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{model_name}.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind_auprc={results['indication']['auprc']} | wall={results['wall_clock_seconds']}s")
    return results
