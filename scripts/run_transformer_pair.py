"""
Q1 controlled pair: TransformerKG vs TransformerNoKG.

Same backbone. Same hyperparameters. Same seeds. Same splits.
Only difference: KG condition receives neighbor triple embeddings as context.

Results to:
  results/transformer_kg/{split}/seed_{n}/transformer_kg.json
  results/transformer_nokg/{split}/seed_{n}/transformer_nokg.json
"""

import argparse
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Data loading FIRST — no torch/pyg at module level.
from src.data.primekg_loader import load_primekg
from src.data.splits import build_splits
from src.evaluation.metrics import auprc as _auprc, auroc as _auroc

SPLITS_DIR = ROOT / "data" / "splits"
SEEDS = [42, 0, 1]
SPLITS = ["standard", "zeroshot"]
THERAPEUTIC_RELATIONS = {"indication", "contraindication"}


def check_splits_exist(seed):
    for split in SPLITS:
        for subset in ["train", "val", "test"]:
            if not (SPLITS_DIR / split / f"seed_{seed}" / f"{subset}.csv").exists():
                return False
    return True


def train_transformer(
    model,
    train_pairs: dict,
    val_pairs: dict,
    test_edges: pd.DataFrame,
    entity_index: dict,
    n_diseases: int,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    neg_ratio: int = 5,
    mixed_precision: bool = True,
    patience: int = 10,
    model_name: str = "transformer_nokg",
    split_name: str = "standard",
    seed: int = 42,
    results_dir: Path = None,
) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.amp import GradScaler, autocast

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=mixed_precision)
    drug_id_map = entity_index.get("drug", {})
    disease_id_map = entity_index.get("disease", {})

    best_val_auprc = -1.0
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        n_tasks = 0

        with autocast("cuda", enabled=mixed_precision):
            for rel, (drug_idx, disease_idx) in train_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                pos_scores = model(drug_idx, disease_idx)
                neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * neg_ratio,), device=device)
                neg_drug = drug_idx.repeat(neg_ratio)
                neg_scores = model(neg_drug, neg_dis)
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                scores = torch.cat([pos_scores, neg_scores])
                total_loss = total_loss + F.binary_cross_entropy_with_logits(scores, labels)
                n_tasks += 1
            if n_tasks:
                total_loss = total_loss / n_tasks

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_auprcs = []
            with torch.no_grad():
                for rel, (drug_idx, disease_idx) in val_pairs.items():
                    drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                    ps = torch.sigmoid(model(drug_idx, disease_idx)).cpu().numpy()
                    nd = torch.randint(0, n_diseases, (len(drug_idx) * 5,), device=device)
                    ns = torch.sigmoid(model(drug_idx.repeat(5), nd)).cpu().numpy()
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
    test_metrics = {}
    with torch.no_grad():
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
            ps = torch.sigmoid(model(d_idx, dis_idx)).cpu().numpy()
            nd = torch.randint(0, n_diseases, (len(d_idx) * 5,), device=device)
            ns = torch.sigmoid(model(d_idx.repeat(5), nd)).cpu().numpy()
            yt = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
            ys = np.concatenate([ps, ns])
            test_metrics[rel] = {"auprc": _auprc(yt, ys), "auroc": _auroc(yt, ys),
                                 "n_pairs": len(yt)}

    use_kg = getattr(model, "use_kg", None)
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
        "use_kg": use_kg,
        "indication": test_metrics.get("indication", {"auprc": None, "auroc": None}),
        "contraindication": test_metrics.get("contraindication", {"auprc": None, "auroc": None}),
        "notes": f"best_val_auprc={best_val_auprc:.4f}",
    }

    if results_dir:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{model_name}.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind_auprc={results['indication'].get('auprc')} | "
          f"contra_auprc={results['contraindication'].get('auprc')}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Step 1: load data before torch
    print("[setup] Loading PrimeKG (reading CSV before torch import) ...")
    kg, entity_index, _ = load_primekg()

    for seed in args.seeds:
        if not check_splits_exist(seed):
            build_splits(kg, seeds=[seed])

    # Step 2: safe to import torch now
    import torch
    from src.utils.gpu_config import setup_gpu
    from src.models.transformer_kg import TransformerKG
    from src.models.transformer_nokg import TransformerNoKG

    n_drugs = len(entity_index.get("drug", {}))
    n_diseases = len(entity_index.get("disease", {}))
    all_relations = kg["relation"].unique().tolist()
    rel_index = {r: i for i, r in enumerate(all_relations)}
    drug_id_map = entity_index.get("drug", {})
    disease_id_map = entity_index.get("disease", {})

    device = setup_gpu(0)
    splits_to_run = SPLITS if args.split == "both" else [args.split]

    for seed in args.seeds:
        for split in splits_to_run:
            print(f"\n{'='*60}")
            print(f"[run] transformer pair | split={split} | seed={seed}")
            torch.manual_seed(seed)

            split_dir = SPLITS_DIR / split / f"seed_{seed}"
            train_edges = pd.read_csv(split_dir / "train.csv")
            val_edges = pd.read_csv(split_dir / "val.csv")
            test_edges = pd.read_csv(split_dir / "test.csv")

            def _edge_tensors(df):
                pairs = {}
                for rel, grp in df.groupby("relation"):
                    if rel not in THERAPEUTIC_RELATIONS:
                        continue
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

            # KG condition
            model_kg = TransformerKG(
                n_drugs=n_drugs, n_diseases=n_diseases,
                n_relation_types=len(rel_index),
                hidden_dim=args.hidden_dim,
            )
            train_transformer(
                model=model_kg, train_pairs=train_pairs, val_pairs=val_pairs,
                test_edges=test_edges, entity_index=entity_index,
                n_diseases=n_diseases, device=device, epochs=args.epochs, lr=args.lr,
                model_name="transformer_kg", split_name=split, seed=seed,
                results_dir=ROOT / "results" / "transformer_kg",
            )

            # No-KG condition (reset to same seed)
            torch.manual_seed(seed)
            model_nokg = TransformerNoKG(
                n_drugs=n_drugs, n_diseases=n_diseases,
                hidden_dim=args.hidden_dim,
            )
            train_transformer(
                model=model_nokg, train_pairs=train_pairs, val_pairs=val_pairs,
                test_edges=test_edges, entity_index=entity_index,
                n_diseases=n_diseases, device=device, epochs=args.epochs, lr=args.lr,
                model_name="transformer_nokg", split_name=split, seed=seed,
                results_dir=ROOT / "results" / "transformer_nokg",
            )

    print("\n[done] Transformer pair complete.")


if __name__ == "__main__":
    main()
