"""
Transformer + KG context (Q1 KG-augmented condition).

Same backbone as transformer_nokg.py — a small 2-layer transformer encoder.
Difference: input to the transformer includes neighbor triple embeddings
retrieved from the KG for each drug and disease.

KG augmentation strategy:
  For each (drug, disease) pair, retrieve the top-K triples involving
  each entity from the KG (neighbor edges). Embed each triple as
  (subject_emb, relation_emb, object_emb) concatenated, project to D,
  then prepend these K context tokens to the sequence fed to the transformer.

This gives the transformer structural KG context without building a GNN.
It is a valid controlled alternative to the GNN approach and directly
tests whether a non-GNN model benefits from KG information.

Everything NOT in this file (architecture depth, hidden dim, lr, seeds)
is held identical to transformer_nokg.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleEmbedder(nn.Module):
    """
    Embeds a set of neighbor triples (h, r, t) as context tokens.
    Each triple -> one D-dimensional token.
    """

    def __init__(
        self,
        n_relation_types: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(n_relation_types, hidden_dim)
        # Project (h_emb || r_emb || t_emb) -> hidden_dim
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self,
        head_embs: torch.Tensor,   # [B, K, D]
        rel_ids: torch.Tensor,     # [B, K]
        tail_embs: torch.Tensor,   # [B, K, D]
    ) -> torch.Tensor:
        r = self.rel_emb(rel_ids)  # [B, K, D]
        x = torch.cat([head_embs, r, tail_embs], dim=-1)  # [B, K, 3D]
        return self.proj(x)  # [B, K, D]


class TransformerKG(nn.Module):
    """
    Controlled KG-augmented condition for Q1.
    Backbone: 2-layer transformer encoder.
    Input: entity embedding + K neighbor triple embeddings as context tokens.
    """

    def __init__(
        self,
        n_drugs: int,
        n_diseases: int,
        n_relation_types: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_neighbors: int = 8,  # K context triples per entity
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.use_kg = True  # marker for ablation tracking

        # Entity embeddings
        self.drug_emb = nn.Embedding(n_drugs, hidden_dim)
        self.disease_emb = nn.Embedding(n_diseases, hidden_dim)

        # KG neighbor triple embedder
        self.triple_embedder = TripleEmbedder(n_relation_types, hidden_dim)

        # CLS-like type tokens to distinguish drug vs disease in sequence
        self.drug_type_token = nn.Parameter(torch.randn(hidden_dim))
        self.disease_type_token = nn.Parameter(torch.randn(hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.score_head = nn.Linear(hidden_dim * 2, 1)

    def _get_entity_emb(
        self,
        entity_type: str,
        entity_idx: torch.Tensor,
    ) -> torch.Tensor:
        if entity_type == "drug":
            return self.drug_emb(entity_idx)
        return self.disease_emb(entity_idx)

    def encode_with_context(
        self,
        entity_emb: torch.Tensor,              # [B, D]
        type_token: torch.Tensor,              # [D]
        neighbor_head_embs: torch.Tensor,      # [B, K, D] or None
        neighbor_rel_ids: torch.Tensor,        # [B, K] or None
        neighbor_tail_embs: torch.Tensor,      # [B, K, D] or None
    ) -> torch.Tensor:
        """
        Builds a sequence: [entity_token, context_0, ..., context_K]
        and passes it through the transformer. Returns the entity token output.
        """
        B, D = entity_emb.shape
        # Entity token: entity embedding + type token
        entity_token = entity_emb + type_token.unsqueeze(0)  # [B, D]
        seq = entity_token.unsqueeze(1)  # [B, 1, D]

        if neighbor_head_embs is not None and neighbor_head_embs.size(1) > 0:
            ctx = self.triple_embedder(
                neighbor_head_embs, neighbor_rel_ids, neighbor_tail_embs
            )  # [B, K, D]
            seq = torch.cat([seq, ctx], dim=1)  # [B, 1+K, D]

        out = self.transformer(seq)  # [B, 1+K, D]
        return out[:, 0, :]  # return CLS-position (entity token output)

    def forward(
        self,
        drug_idx: torch.Tensor,             # [B]
        disease_idx: torch.Tensor,          # [B]
        drug_neighbor_data: dict | None = None,     # {head, rel, tail}
        disease_neighbor_data: dict | None = None,
    ) -> torch.Tensor:
        """
        Returns [B] scores.
        """
        drug_base = self.drug_emb(drug_idx)        # [B, D]
        disease_base = self.disease_emb(disease_idx)

        drug_enc = self.encode_with_context(
            drug_base, self.drug_type_token,
            drug_neighbor_data.get("head") if drug_neighbor_data else None,
            drug_neighbor_data.get("rel") if drug_neighbor_data else None,
            drug_neighbor_data.get("tail") if drug_neighbor_data else None,
        )

        disease_enc = self.encode_with_context(
            disease_base, self.disease_type_token,
            disease_neighbor_data.get("head") if disease_neighbor_data else None,
            disease_neighbor_data.get("rel") if disease_neighbor_data else None,
            disease_neighbor_data.get("tail") if disease_neighbor_data else None,
        )

        pair_rep = torch.cat([drug_enc, disease_enc], dim=-1)  # [B, 2D]
        return self.score_head(pair_rep).squeeze(-1)  # [B]


def build_neighbor_lookup(
    kg: "pd.DataFrame",
    entity_index: dict,
    relation_index: dict,
    max_neighbors: int = 8,
    drug_node_type: str = "drug",
    disease_node_type: str = "disease",
) -> tuple[dict, dict]:
    """
    Precomputes neighbor triple indices for each drug and disease.
    Returns:
      drug_neighbors:    {entity_local_idx: [(head_idx, rel_id, tail_idx), ...]}
      disease_neighbors: same
    """
    drug_nbrs: dict = {}
    disease_nbrs: dict = {}

    drug_id_map = entity_index.get(drug_node_type, {})
    disease_id_map = entity_index.get(disease_node_type, {})

    for _, row in kg.iterrows():
        x_type, x_id = row["x_type"], row["x_id"]
        y_type, y_id = row["y_type"], row["y_id"]
        rel = row["relation"]

        if rel not in relation_index:
            continue
        rel_id = relation_index[rel]

        if x_type == drug_node_type and x_id in drug_id_map:
            src = drug_id_map[x_id]
            if src not in drug_nbrs:
                drug_nbrs[src] = []
            if len(drug_nbrs[src]) < max_neighbors:
                dst_map = entity_index.get(y_type, {})
                if y_id in dst_map:
                    drug_nbrs[src].append((drug_id_map[x_id], rel_id, dst_map[y_id]))

        if y_type == disease_node_type and y_id in disease_id_map:
            dst = disease_id_map[y_id]
            if dst not in disease_nbrs:
                disease_nbrs[dst] = []
            if len(disease_nbrs[dst]) < max_neighbors:
                src_map = entity_index.get(x_type, {})
                if x_id in src_map:
                    disease_nbrs[dst].append((src_map[x_id], rel_id, disease_id_map[y_id]))

    return drug_nbrs, disease_nbrs
