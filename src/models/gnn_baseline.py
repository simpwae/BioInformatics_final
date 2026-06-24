"""
Plain GNN baseline (no disease-similarity module, no metric learning).
This is the "GNN" column in the Q5 comparison table.

Architecture: Heterogeneous Graph Transformer (HGT) — two layers.

Node features: learnable nn.Embedding per node type (Xavier uniform init).
Using raw node IDs as float features causes NaN due to large unnormalized inputs;
embedding tables are the standard approach for KG learning without external features.

This model has NO attention ablation flag — that lives in txgnn.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class GNNBaseline(nn.Module):
    """
    Two-layer HGT over PrimeKG.
    Scoring: dot product between drug embedding and disease embedding.
    """

    def __init__(
        self,
        metadata: tuple,          # (node_types, edge_types) from HeteroData.metadata()
        node_counts: dict,        # {node_type: n_nodes} from heterodata
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim

        # Learnable embedding per node type — Xavier uniform initialization
        self.node_embs = nn.ModuleDict({
            nt: nn.Embedding(node_counts[nt], hidden_dim) for nt in node_types
        })
        for emb in self.node_embs.values():
            nn.init.xavier_uniform_(emb.weight)

        # HGT convolution layers
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, num_heads)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def encode(self, edge_index_dict: dict) -> dict:
        """Returns per-node-type embeddings after GNN layers.

        Uses the full embedding table for each node type (full-graph training).
        For mini-batch training, pass a node_id tensor to index into the table.
        """
        h = {nt: self.dropout(F.relu(self.node_embs[nt].weight))
             for nt in self.node_types}

        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {nt: self.dropout(F.relu(emb)) for nt, emb in h.items()}

        return {nt: self.out_norm(emb) for nt, emb in h.items()}

    def score(self, drug_emb: torch.Tensor, disease_emb: torch.Tensor) -> torch.Tensor:
        """Dot product score for a batch of (drug, disease) pairs."""
        return (drug_emb * disease_emb).sum(dim=-1)

    def forward(
        self,
        edge_index_dict: dict,
        drug_idx: torch.Tensor,    # [B] indices into drug node embeddings
        disease_idx: torch.Tensor, # [B] indices into disease node embeddings
        drug_node_type: str = "drug",
        disease_node_type: str = "disease",
    ) -> torch.Tensor:
        h = self.encode(edge_index_dict)
        return self.score(h[drug_node_type][drug_idx], h[disease_node_type][disease_idx])
