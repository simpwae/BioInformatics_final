"""
Phase 5 — Alternative B: End-to-end joint contrastive training.

Folds the disease-similarity objective (metric learning) into one unified
training stage alongside the therapeutic task loss, with no KG pretraining.

Loss = L_therapeutic + beta * L_contrastive_similarity

L_contrastive_similarity: InfoNCE-style loss that pulls together diseases
sharing at least one approved drug and pushes apart diseases with no overlap.

This directly tests whether the metric-learning signal that enables zero-shot
transfer can be learned jointly with the task, without Phase 1 pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class JointContrastiveModel(nn.Module):
    """
    GNN trained end-to-end with therapeutic loss + InfoNCE disease-similarity loss.
    No separate pretraining phase.
    """

    def __init__(
        self,
        metadata: tuple,
        node_counts: dict,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_neighbors: int = 5,
        contrastive_weight: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        self.node_embs = nn.ModuleDict({
            nt: nn.Embedding(node_counts[nt], hidden_dim) for nt in node_types
        })
        for emb in self.node_embs.values():
            nn.init.xavier_uniform_(emb.weight)

        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, num_heads)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def encode(self, edge_index_dict):
        h = {nt: self.dropout(F.relu(self.node_embs[nt].weight))
             for nt in self.node_types}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {nt: self.dropout(F.relu(emb)) for nt, emb in h.items()}
        return {nt: self.out_norm(emb) for nt, emb in h.items()}

    def infonce_disease_loss(
        self,
        disease_embs: torch.Tensor,
        positive_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE loss: for each positive pair (di, dj), negatives are all other
        diseases in the batch. NT-Xent formulation.
        """
        if positive_pairs.size(0) == 0:
            return torch.tensor(0.0, device=disease_embs.device)

        z = F.normalize(self.contrastive_proj(disease_embs), dim=-1)

        unique_idx = positive_pairs.flatten().unique()
        z_sub = z[unique_idx]

        idx_remap = {int(v): i for i, v in enumerate(unique_idx)}
        i_idx = torch.tensor([idx_remap[int(p[0])] for p in positive_pairs],
                              device=disease_embs.device)
        j_idx = torch.tensor([idx_remap[int(p[1])] for p in positive_pairs],
                              device=disease_embs.device)

        sim = torch.mm(z_sub, z_sub.t()) / self.temperature
        sim.fill_diagonal_(float("-inf"))

        loss_i = F.cross_entropy(sim[i_idx], j_idx)
        loss_j = F.cross_entropy(sim[j_idx], i_idx)
        return (loss_i + loss_j) / 2

    def zero_shot_score(
        self,
        drug_embs: torch.Tensor,
        query_disease_emb: torch.Tensor,
        support_disease_embs: torch.Tensor,
    ) -> torch.Tensor:
        q = F.normalize(self.contrastive_proj(query_disease_emb), dim=-1)
        s = F.normalize(self.contrastive_proj(support_disease_embs), dim=-1)
        sim = torch.mm(q, s.t())
        topk_sims, topk_idx = sim.topk(min(self.n_neighbors, sim.size(1)), dim=-1)
        weights = F.softmax(topk_sims, dim=-1)
        support_scores = torch.mm(support_disease_embs, drug_embs.t())
        top_scores = support_scores[topk_idx]
        return (weights.unsqueeze(-1) * top_scores).sum(dim=1)

    def forward(self, edge_index_dict, drug_idx, disease_idx,
                drug_node_type="drug", disease_node_type="disease"):
        h = self.encode(edge_index_dict)
        return (h[drug_node_type][drug_idx] * h[disease_node_type][disease_idx]).sum(-1)

    def model_config(self):
        return {
            "architecture": "joint_contrastive",
            "hidden_dim": self.hidden_dim,
            "contrastive_weight": self.contrastive_weight,
            "temperature": self.temperature,
            "n_neighbors": self.n_neighbors,
            "reproduction_type": "original_ablation",
        }
