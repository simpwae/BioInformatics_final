"""
TxGNN: scaled reproduction of Huang et al. (2024), Nature Medicine.
DOI: 10.1038/s41591-024-03233-x

Scaled from the published setup to fit RTX 4060 (8 GB VRAM).
Label all results from this model as reproduction_type="scaled_reproduction".

Architecture (from GitHub README and paper description):
  - HGT-style GNN encoder with optional attention (toggled by use_attention flag)
  - Disease-similarity metric-learning module for zero-shot transfer
  - Two-phase training:
      Phase 1 (pretrain): self-supervised link prediction over all edge types
      Phase 2 (finetune): therapeutic task with metric-learning loss

Changes from published setup (record every deviation):
  1. hidden_dim reduced from 512 to 64 (VRAM constraint — 128 still OOMs on RTX 4060 8 GB)
  2. Number of GNN layers: 3 -> 2 (VRAM constraint)
  3. num_heads: 8 -> 4 (VRAM constraint)
  4. Node features: learnable nn.Embedding per node type (paper uses pre-trained features)
  5. anatomy_protein_present and drug_drug edges excluded from message passing (VRAM constraint)
  6. Disease similarity computed over all training diseases (same as paper)
  7. Batch construction: same negative sampling strategy (corrupt disease)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class DiseaseSimilarityModule(nn.Module):
    """
    Metric-learning module for zero-shot disease transfer.

    For a query disease with no training edges, this module finds the K most
    similar seen diseases (by learned embedding similarity) and transfers their
    drug predictions to the query.

    This is the component that enables zero-shot repurposing. Without it, the
    model cannot score drugs for unseen diseases.
    """

    def __init__(self, hidden_dim: int, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors
        # Projection to similarity space (learnable)
        self.sim_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def compute_similarity(
        self, query_emb: torch.Tensor, support_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Cosine similarity between query and all support (training) diseases.
        query_emb:   [B, D]
        support_embs: [N_support, D]
        Returns:     [B, N_support]
        """
        q = F.normalize(self.sim_proj(query_emb), dim=-1)
        s = F.normalize(self.sim_proj(support_embs), dim=-1)
        return torch.mm(q, s.t())  # [B, N_support]

    def transfer_drug_scores(
        self,
        query_disease_emb: torch.Tensor,   # [B, D]
        support_disease_embs: torch.Tensor, # [N_support, D]
        drug_embs: torch.Tensor,            # [N_drugs, D]
        drug_node_type: str = "drug",
    ) -> torch.Tensor:
        """
        For each query disease, computes weighted-average drug scores from
        the K nearest support diseases.
        Returns: [B, N_drugs] score matrix.
        """
        sim = self.compute_similarity(query_disease_emb, support_disease_embs)
        topk_sims, topk_idx = sim.topk(
            min(self.n_neighbors, sim.size(1)), dim=-1
        )  # [B, K]
        topk_weights = F.softmax(topk_sims, dim=-1)  # [B, K]

        # For each support disease, compute drug scores via dot product
        # support_disease_embs: [N_support, D], drug_embs: [N_drugs, D]
        # -> [N_support, N_drugs]
        support_scores = torch.mm(support_disease_embs, drug_embs.t())

        # Gather top-K support disease scores: [B, K, N_drugs]
        top_support_scores = support_scores[topk_idx]  # [B, K, N_drugs]

        # Weighted average: [B, N_drugs]
        transferred = (topk_weights.unsqueeze(-1) * top_support_scores).sum(dim=1)
        return transferred

    def metric_learning_loss(
        self,
        disease_embs: torch.Tensor,    # [B, D] all training disease embeddings
        pos_pairs: torch.Tensor,       # [N_pos, 2] indices of disease pairs that share drugs
        neg_pairs: torch.Tensor,       # [N_neg, 2] indices of disease pairs with no overlap
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Triplet-style loss: similar diseases (sharing drugs) pulled together,
        dissimilar diseases pushed apart in similarity space.
        """
        q = F.normalize(self.sim_proj(disease_embs), dim=-1)

        if pos_pairs.size(0) == 0 or neg_pairs.size(0) == 0:
            return torch.tensor(0.0, device=disease_embs.device)

        pos_sim = (q[pos_pairs[:, 0]] * q[pos_pairs[:, 1]]).sum(-1)
        neg_sim = (q[neg_pairs[:, 0]] * q[neg_pairs[:, 1]]).sum(-1)

        # Contrastive: push pos_sim up, neg_sim down
        loss = F.relu(margin - pos_sim).mean() + F.relu(neg_sim + margin).mean()
        return loss


class TxGNN(nn.Module):
    """
    Scaled TxGNN reproduction.

    Flags that matter for ablations (Phase 6):
      use_attention:      HGT attention vs. mean aggregation (Q6)
      use_similarity:     disease-similarity module on/off (Q6)
    """

    def __init__(
        self,
        metadata: tuple,
        node_counts: dict,              # {node_type: n_nodes} from heterodata
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,     # Q6 ablation flag
        use_similarity: bool = True,    # Q6 ablation flag
        n_neighbors: int = 5,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_similarity = use_similarity

        self.node_embs = nn.ModuleDict({
            nt: nn.Embedding(node_counts[nt], hidden_dim) for nt in node_types
        })
        for emb in self.node_embs.values():
            nn.init.xavier_uniform_(emb.weight)

        if use_attention:
            self.convs = nn.ModuleList([
                HGTConv(hidden_dim, hidden_dim, metadata, num_heads)
                for _ in range(num_layers)
            ])
        else:
            # Mean aggregation: no attention (ablation condition B for Q6)
            from torch_geometric.nn import SAGEConv, to_hetero
            # Use a simpler homogeneous SAGEConv wrapped with to_hetero
            from torch_geometric.nn import HeteroConv, SAGEConv as SAGE
            self.convs = nn.ModuleList([
                HeteroConv(
                    {et: SAGE(hidden_dim, hidden_dim, aggr="mean")
                     for et in edge_types},
                    aggr="mean"
                )
                for _ in range(num_layers)
            ])

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

        if use_similarity:
            self.similarity_module = DiseaseSimilarityModule(hidden_dim, n_neighbors)
        else:
            self.similarity_module = None

    def encode(self, edge_index_dict: dict) -> dict:
        h = {nt: self.dropout(F.relu(self.node_embs[nt].weight))
             for nt in self.node_types}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {nt: self.dropout(F.relu(emb)) for nt, emb in h.items()}
        return {nt: self.out_norm(emb) for nt, emb in h.items()}

    def score_direct(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Dot product score — used for seen diseases."""
        return (drug_emb * disease_emb).sum(dim=-1)

    def score_zero_shot(
        self,
        drug_embs: torch.Tensor,            # [N_drugs, D]
        query_disease_embs: torch.Tensor,   # [B, D]
        support_disease_embs: torch.Tensor, # [N_seen, D]
    ) -> torch.Tensor:
        """
        Zero-shot scoring: transfer from support diseases via similarity.
        Returns [B, N_drugs] scores.
        """
        if self.similarity_module is None:
            # Fallback: direct dot product (similarity module off)
            return torch.mm(query_disease_embs, drug_embs.t())
        return self.similarity_module.transfer_drug_scores(
            query_disease_embs, support_disease_embs, drug_embs
        )

    def forward(
        self,
        edge_index_dict: dict,
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        drug_node_type: str = "drug",
        disease_node_type: str = "disease",
    ) -> torch.Tensor:
        h = self.encode(edge_index_dict)
        return self.score_direct(h[drug_node_type][drug_idx], h[disease_node_type][disease_idx])

    def model_config(self) -> dict:
        """Returns the exact config used — written to results JSON."""
        return {
            "hidden_dim": self.hidden_dim,
            "use_attention": self.use_attention,
            "use_similarity": self.use_similarity,
            "num_layers": len(self.convs),
            "scaled_from_paper": True,
            "paper_hidden_dim": 512,
            "paper_num_layers": 3,
            "paper_num_heads": 8,
            "deviation_reason": "RTX 4060 8GB VRAM constraint",
        }
