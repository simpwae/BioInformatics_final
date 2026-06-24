"""
Phase 5 — Alternative A: Single-stage multi-task training.

No separate pretrain phase. KG link prediction loss and therapeutic task
loss are combined from epoch 1.

Loss = alpha * L_kg_lp + (1 - alpha) * L_therapeutic

This tests Q2: does separating Phase 1 and Phase 2 actually matter, or
can a single-stage curriculum-free approach match it?

Architecture: same HGT encoder as TxGNN. Disease similarity module included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from src.models.txgnn import DiseaseSimilarityModule


class SingleStageModel(nn.Module):
    """
    GNN + disease-similarity, trained single-stage (no Phase 1/Phase 2 split).
    """

    def __init__(
        self,
        metadata: tuple,
        node_counts: dict,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_similarity: bool = True,
        n_neighbors: int = 5,
        kg_loss_weight: float = 0.5,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.kg_loss_weight = kg_loss_weight

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

        self.similarity_module = DiseaseSimilarityModule(hidden_dim, n_neighbors) \
            if use_similarity else None

    def encode(self, edge_index_dict):
        h = {nt: self.dropout(F.relu(self.node_embs[nt].weight))
             for nt in self.node_types}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {nt: self.dropout(F.relu(emb)) for nt, emb in h.items()}
        return {nt: self.out_norm(emb) for nt, emb in h.items()}

    def score(self, drug_emb, disease_emb):
        return (drug_emb * disease_emb).sum(-1)

    def forward(self, edge_index_dict, drug_idx, disease_idx,
                drug_node_type="drug", disease_node_type="disease"):
        h = self.encode(edge_index_dict)
        return self.score(h[drug_node_type][drug_idx], h[disease_node_type][disease_idx])

    def model_config(self):
        return {
            "architecture": "single_stage",
            "hidden_dim": self.hidden_dim,
            "kg_loss_weight": self.kg_loss_weight,
            "use_similarity": self.similarity_module is not None,
            "reproduction_type": "original_ablation",
        }
