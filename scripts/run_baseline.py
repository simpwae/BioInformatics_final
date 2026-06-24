"""
Trains the plain GNN baseline on both splits for all seeds.

Usage:
    python scripts/run_baseline.py [--split standard|zeroshot|both] [--seeds 42 0 1]

Writes to:
    results/gnn/{split}/seed_{n}/{model_name}.json
"""

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Data loading imports only — no torch/pyg at module level.
# load_primekg reads the kg.csv BEFORE torch_geometric is imported.
# Importing torch_geometric before reading a large CSV causes a Windows access violation.
from src.data.primekg_loader import load_primekg
from src.data.splits import build_splits

RESULTS_DIR = ROOT / "results" / "gnn"
SPLITS_DIR = ROOT / "data" / "splits"

SEEDS = [42, 0, 1]
SPLITS = ["standard", "zeroshot"]


def check_splits_exist(seed: int) -> bool:
    for split in SPLITS:
        for subset in ["train", "val", "test"]:
            p = SPLITS_DIR / split / f"seed_{seed}" / f"{subset}.csv"
            if not p.exists():
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2,
                        help="GNN depth. 0 = no message passing (no-KG condition for Q1).")
    parser.add_argument("--model_name", type=str, default="gnn_baseline",
                        help="Label used for result filenames. Use gnn_kg or gnn_no_kg for Q1.")
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Step 1: load PrimeKG data (reads 982 MB CSV via pure pandas)
    # Must happen BEFORE importing torch/pyg — importing pyg before pandas CSV read
    # causes a Windows access violation (torch C extensions corrupt the allocator).
    print("[setup] Loading PrimeKG (reading CSV first, before torch import) ...")
    kg, entity_index, heterodata = load_primekg()

    # Build splits if not already on disk
    for seed in args.seeds:
        if not check_splits_exist(seed):
            print(f"[setup] Splits for seed={seed} not found — building now ...")
            build_splits(kg, seeds=[seed])

    # Step 2: NOW it is safe to import torch and models
    import torch
    from src.utils.gpu_config import setup_gpu
    from src.models.gnn_baseline import GNNBaseline
    from src.training.pretrain import pretrain
    from src.training.finetune import finetune

    device = setup_gpu(0)

    splits_to_run = SPLITS if args.split == "both" else [args.split]

    for seed in args.seeds:
        for split in splits_to_run:
            print(f"\n{'='*60}")
            print(f"[run] GNN baseline | split={split} | seed={seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)

            node_counts = {nt: heterodata[nt].num_nodes for nt in heterodata.node_types}
            model = GNNBaseline(
                metadata=heterodata.metadata(),
                node_counts=node_counts,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
            )
            model_name = args.model_name

            # Phase 1: self-supervised pretraining
            pretrain_log = pretrain(
                model=model,
                data=heterodata,
                device=device,
                epochs=args.pretrain_epochs,
                lr=args.lr,
                results_dir=RESULTS_DIR,
                model_name=model_name,
                seed=seed,
            )

            # Phase 2: fine-tune on therapeutic edges
            split_dir = SPLITS_DIR / split / f"seed_{seed}"
            results = finetune(
                model=model,
                data=heterodata,
                entity_index=entity_index,
                split_dir=split_dir,
                device=device,
                epochs=args.finetune_epochs,
                lr=args.lr,
                results_dir=RESULTS_DIR,
                model_name=model_name,
                split_name=split,
                seed=seed,
            )

            print(f"\n[result] seed={seed} split={split}")
            print(json.dumps(results, indent=2))

    print("\n[done] All runs complete.")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
