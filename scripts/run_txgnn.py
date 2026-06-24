"""
Runs TxGNN (scaled reproduction) on both splits for all seeds.
Phase 1: self-supervised pretrain (reuses src/training/pretrain.py).
Phase 2: fine-tune with disease-similarity loss (src/training/txgnn_train.py).

Usage:
    python scripts/run_txgnn.py [--split standard|zeroshot|both] [--seeds 42 0 1]
    python scripts/run_txgnn.py --model txgnn_no_attn   (Q6 ablation B)
    python scripts/run_txgnn.py --model txgnn_no_sim    (Q6 ablation: sim off)
"""

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Data loading FIRST — no torch/pyg at module level.
# Importing pyg before pandas reads a large CSV causes a Windows access violation.
from src.data.primekg_loader import load_primekg
from src.data.splits import build_splits
from src.data.leakage_check import check_leakage

RESULTS_DIR = ROOT / "results" / "txgnn"
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
    parser.add_argument("--model", default="txgnn",
                        choices=["txgnn", "txgnn_no_attn", "txgnn_no_sim", "txgnn_no_both"])
    parser.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sim_loss_weight", type=float, default=0.3)
    parser.add_argument("--skip_leakage_check", action="store_true")
    args = parser.parse_args()

    # Step 1: load PrimeKG (reads CSV before torch import)
    print("[setup] Loading PrimeKG (reading CSV before torch import) ...")
    kg, entity_index, heterodata = load_primekg()

    for seed in args.seeds:
        if not check_splits_exist(seed):
            print(f"[setup] Splits for seed={seed} not found — building ...")
            build_splits(kg, seeds=[seed])

    # Leakage check gates zero-shot results
    if not args.skip_leakage_check and ("zeroshot" in args.split or args.split == "both"):
        for seed in args.seeds:
            result = check_leakage(seed)
            if result["status"] == "FAIL":
                print(f"[ABORT] Leakage detected in zero-shot split seed={seed}. "
                      f"Fix split. See results/metrics/leakage_check_seed{seed}.json")
                sys.exit(1)
        print("[leakage] All zero-shot splits PASS leakage check.")

    # Step 2: NOW safe to import torch and models
    import torch
    from src.utils.gpu_config import setup_gpu
    from src.models.txgnn import TxGNN
    from src.models.txgnn_no_attn import build_txgnn_no_attn, build_txgnn_no_sim, build_txgnn_no_both
    from src.training.pretrain import pretrain
    from src.training.txgnn_train import finetune_txgnn

    node_counts = {nt: heterodata[nt].num_nodes for nt in heterodata.node_types}

    MODEL_BUILDERS = {
        "txgnn":         lambda nc, h: TxGNN(heterodata.metadata(), nc, hidden_dim=h,
                                              use_attention=True, use_similarity=True),
        "txgnn_no_attn": lambda nc, h: build_txgnn_no_attn(heterodata.metadata(), nc, hidden_dim=h),
        "txgnn_no_sim":  lambda nc, h: build_txgnn_no_sim(heterodata.metadata(), nc, hidden_dim=h),
        "txgnn_no_both": lambda nc, h: build_txgnn_no_both(heterodata.metadata(), nc, hidden_dim=h),
    }

    device = setup_gpu(0)
    splits_to_run = SPLITS if args.split == "both" else [args.split]
    results_dir = ROOT / "results" / args.model

    for seed in args.seeds:
        for split in splits_to_run:
            print(f"\n{'='*60}")
            print(f"[run] {args.model} | split={split} | seed={seed}")
            print(f"{'='*60}")
            torch.manual_seed(seed)

            model = MODEL_BUILDERS[args.model](node_counts, args.hidden_dim)

            # Phase 1: self-supervised pretraining
            pretrain(
                model=model,
                data=heterodata,
                device=device,
                epochs=args.pretrain_epochs,
                lr=args.lr,
                results_dir=results_dir,
                model_name=args.model,
                seed=seed,
            )

            # Phase 2: therapeutic fine-tune + disease similarity
            split_dir = SPLITS_DIR / split / f"seed_{seed}"
            results = finetune_txgnn(
                model=model,
                data=heterodata,
                entity_index=entity_index,
                split_dir=split_dir,
                device=device,
                epochs=args.finetune_epochs,
                lr=args.lr,
                results_dir=results_dir,
                model_name=args.model,
                split_name=split,
                seed=seed,
                sim_loss_weight=args.sim_loss_weight,
            )

            print(json.dumps({k: v for k, v in results.items()
                               if k != "per_disease_results"}, indent=2))

    print(f"\n[done] Results in: {results_dir}")


if __name__ == "__main__":
    main()
