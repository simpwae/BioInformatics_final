"""
Phase 1 training: self-supervised link prediction over all PrimeKG edge types.
Used by TxGNN and alternative models that follow the two-phase protocol.
The GNN baseline uses this for pre-training too (to ensure fair comparison),
but without the metric-learning module added in Phase 2.
"""

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch_geometric.data import HeteroData


def negative_sample(
    edge_index: torch.Tensor,
    num_nodes_src: int,
    num_nodes_dst: int,
    device: torch.device,
    neg_ratio: int = 1,
) -> torch.Tensor:
    """
    Uniform random negative sampling. Returns [2, N*neg_ratio] corrupted edges
    (corrupt the tail node only).
    """
    n_pos = edge_index.size(1)
    neg_dst = torch.randint(0, num_nodes_dst, (n_pos * neg_ratio,), device=device)
    neg_src = edge_index[0].repeat(neg_ratio)
    return torch.stack([neg_src, neg_dst], dim=0)


def link_prediction_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy link prediction loss."""
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    return F.binary_cross_entropy_with_logits(scores, labels)


def pretrain(
    model: torch.nn.Module,
    data: HeteroData,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    mixed_precision: bool = True,
    results_dir: Path = None,
    model_name: str = "gnn_baseline",
    seed: int = 42,
) -> dict:
    """
    Runs self-supervised link prediction pretraining over all edge types.
    Returns a dict with final training loss logged per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    logs = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        with autocast("cuda", enabled=mixed_precision):
            h = model.encode(edge_index_dict)
            total_loss = torch.tensor(0.0, device=device)
            n_et = 0

            for (src_type, rel, dst_type), ei in edge_index_dict.items():
                if ei.size(1) == 0:
                    continue
                src_emb = h[src_type][ei[0]]
                dst_emb = h[dst_type][ei[1]]
                pos_scores = (src_emb * dst_emb).sum(-1)

                neg_ei = negative_sample(
                    ei,
                    data[src_type].num_nodes,
                    data[dst_type].num_nodes,
                    device,
                )
                neg_src_emb = h[src_type][neg_ei[0]]
                neg_dst_emb = h[dst_type][neg_ei[1]]
                neg_scores = (neg_src_emb * neg_dst_emb).sum(-1)

                total_loss = total_loss + link_prediction_loss(pos_scores, neg_scores)
                n_et += 1

            if n_et > 0:
                total_loss = total_loss / n_et

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = float(total_loss.item())
        elapsed = time.time() - t0
        logs.append({"epoch": epoch, "loss": loss_val, "elapsed_s": round(elapsed, 1)})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [pretrain] epoch={epoch}/{epochs}  loss={loss_val:.4f}  t={elapsed:.0f}s")

    pretrain_log = {
        "model": model_name,
        "phase": "pretrain",
        "seed": seed,
        "epochs": epochs,
        "final_loss": logs[-1]["loss"],
        "total_wall_s": logs[-1]["elapsed_s"],
        "epoch_log": logs,
    }

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / f"{model_name}_pretrain_seed{seed}.json"
        with open(log_path, "w") as f:
            json.dump(pretrain_log, f, indent=2)
        print(f"  [pretrain] log -> {log_path}")

    return pretrain_log
