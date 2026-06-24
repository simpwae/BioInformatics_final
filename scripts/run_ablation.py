"""
Phase 6: Ablation runs.

Ablation matrix (rows = components, columns = split):
  - txgnn              (attention=ON,  similarity=ON)   <- full model
  - txgnn_no_attn      (attention=OFF, similarity=ON)   <- Q6 condition B
  - txgnn_no_sim       (attention=ON,  similarity=OFF)  <- shows sim is load-bearing
  - txgnn_no_both      (attention=OFF, similarity=OFF)  <- plain GNN equivalent

Q6 decision rule (from CLAUDE.md, set BEFORE running):
  delta = auprc(txgnn) - auprc(txgnn_no_attn) on zero-shot split
  delta < 0.02  -> attention optional
  delta >= 0.02 -> attention matters; report that

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --variant txgnn_no_attn --seeds 42
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

SPLITS_DIR = ROOT / "data" / "splits"
SEEDS = [42, 0, 1]
SPLITS = ["standard", "zeroshot"]
ABLATION_VARIANT_NAMES = ["txgnn", "txgnn_no_attn", "txgnn_no_sim", "txgnn_no_both"]


def check_splits_exist(seed: int) -> bool:
    for split in SPLITS:
        for subset in ["train", "val", "test"]:
            if not (SPLITS_DIR / split / f"seed_{seed}" / f"{subset}.csv").exists():
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all",
                        choices=ABLATION_VARIANT_NAMES + ["all"])
    parser.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Step 1: load data before torch
    print("[setup] Loading PrimeKG (reading CSV before torch import) ...")
    kg, entity_index, heterodata = load_primekg()

    for seed in args.seeds:
        if not check_splits_exist(seed):
            build_splits(kg, seeds=[seed])

    # Step 2: safe to import torch now
    import torch
    from src.utils.gpu_config import setup_gpu
    from src.models.txgnn import TxGNN
    from src.models.txgnn_no_attn import build_txgnn_no_attn, build_txgnn_no_sim, build_txgnn_no_both
    from src.training.pretrain import pretrain
    from src.training.txgnn_train import finetune_txgnn

    node_counts = {nt: heterodata[nt].num_nodes for nt in heterodata.node_types}

    def make_model(variant, node_counts, hidden_dim):
        meta = heterodata.metadata()
        if variant == "txgnn":
            return TxGNN(meta, node_counts, hidden_dim=hidden_dim,
                         use_attention=True, use_similarity=True)
        elif variant == "txgnn_no_attn":
            return build_txgnn_no_attn(meta, node_counts, hidden_dim=hidden_dim)
        elif variant == "txgnn_no_sim":
            return build_txgnn_no_sim(meta, node_counts, hidden_dim=hidden_dim)
        elif variant == "txgnn_no_both":
            return build_txgnn_no_both(meta, node_counts, hidden_dim=hidden_dim)

    device = setup_gpu(0)
    variants = ABLATION_VARIANT_NAMES if args.variant == "all" else [args.variant]
    splits_to_run = SPLITS if args.split == "both" else [args.split]

    for variant in variants:
        results_dir = ROOT / "results" / "ablations" / variant
        for seed in args.seeds:
            for split in splits_to_run:
                print(f"\n{'='*60}")
                print(f"[ablation] {variant} | split={split} | seed={seed}")
                torch.manual_seed(seed)

                model = make_model(variant, node_counts, args.hidden_dim)

                pretrain(
                    model=model, data=heterodata, device=device,
                    epochs=args.pretrain_epochs, lr=args.lr,
                    results_dir=results_dir, model_name=variant, seed=seed,
                )

                split_dir = SPLITS_DIR / split / f"seed_{seed}"
                finetune_txgnn(
                    model=model, data=heterodata, entity_index=entity_index,
                    split_dir=split_dir, device=device, epochs=args.finetune_epochs,
                    lr=args.lr, results_dir=results_dir,
                    model_name=variant, split_name=split, seed=seed,
                )

    print("\n[done] All ablation variants complete.")
    print("Run: python scripts/build_ablation_matrix.py")


if __name__ == "__main__":
    main()
