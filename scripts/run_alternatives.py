"""
Phase 5: Runs alternative training strategies (Q2).
  - single_stage: KG LP + therapeutic task jointly from epoch 1
  - joint_contrastive: InfoNCE disease similarity + therapeutic task, no pretrain

"Better" is defined before running (from CLAUDE.md):
  Primary: higher AUPRC at equal-or-lower wall-clock seconds.
  Secondary: matching AUPRC at >= 30% wall-clock reduction.

Usage:
    python scripts/run_alternatives.py [--method single_stage|joint_contrastive|both]
"""

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Data loading FIRST — no torch/pyg at module level.
from src.data.primekg_loader import load_primekg
from src.data.splits import build_splits
from src.data.leakage_check import check_leakage

SPLITS_DIR = ROOT / "data" / "splits"
SEEDS = [42, 0, 1]
SPLITS = ["standard", "zeroshot"]


def check_splits_exist(seed: int) -> bool:
    for split in SPLITS:
        for subset in ["train", "val", "test"]:
            if not (SPLITS_DIR / split / f"seed_{seed}" / f"{subset}.csv").exists():
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="both",
                        choices=["single_stage", "joint_contrastive", "both"])
    parser.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip_leakage_check", action="store_true")
    args = parser.parse_args()

    # Step 1: load data before torch
    print("[setup] Loading PrimeKG (reading CSV before torch import) ...")
    kg, entity_index, heterodata = load_primekg()

    for seed in args.seeds:
        if not check_splits_exist(seed):
            build_splits(kg, seeds=[seed])

    if not args.skip_leakage_check:
        for seed in args.seeds:
            result = check_leakage(seed)
            if result["status"] == "FAIL":
                print(f"[ABORT] Leakage in zeroshot split seed={seed}.")
                sys.exit(1)

    # Step 2: safe to import torch now
    import torch
    from src.utils.gpu_config import setup_gpu
    from src.models.single_stage import SingleStageModel
    from src.models.joint_contrastive import JointContrastiveModel
    from src.training.single_stage_train import train_single_stage, train_joint_contrastive

    node_counts = {nt: heterodata[nt].num_nodes for nt in heterodata.node_types}
    device = setup_gpu(0)

    splits_to_run = SPLITS if args.split == "both" else [args.split]
    methods_to_run = (["single_stage", "joint_contrastive"]
                      if args.method == "both" else [args.method])

    for method in methods_to_run:
        results_dir = ROOT / "results" / f"alt_{method}"
        for seed in args.seeds:
            for split in splits_to_run:
                print(f"\n{'='*60}")
                print(f"[run] {method} | split={split} | seed={seed}")
                torch.manual_seed(seed)

                split_dir = SPLITS_DIR / split / f"seed_{seed}"

                if method == "single_stage":
                    model = SingleStageModel(
                        metadata=heterodata.metadata(),
                        node_counts=node_counts,
                        hidden_dim=args.hidden_dim,
                    )
                    results = train_single_stage(
                        model=model, data=heterodata, entity_index=entity_index,
                        split_dir=split_dir, device=device, epochs=args.epochs,
                        lr=args.lr, results_dir=results_dir,
                        model_name=method, split_name=split, seed=seed,
                    )
                elif method == "joint_contrastive":
                    model = JointContrastiveModel(
                        metadata=heterodata.metadata(),
                        node_counts=node_counts,
                        hidden_dim=args.hidden_dim,
                    )
                    results = train_joint_contrastive(
                        model=model, data=heterodata, entity_index=entity_index,
                        split_dir=split_dir, device=device, epochs=args.epochs,
                        lr=args.lr, results_dir=results_dir,
                        model_name=method, split_name=split, seed=seed,
                    )

                print(json.dumps({k: v for k, v in results.items()
                                  if k != "per_disease_results"}, indent=2))

    print("\n[done] All alternatives complete.")


if __name__ == "__main__":
    main()
