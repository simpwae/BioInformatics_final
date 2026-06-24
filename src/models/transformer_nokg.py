"""
Transformer baseline — NO KG context (Q1 no-KG condition).

Identical backbone to transformer_kg.py:
  - Same hidden_dim, nhead, num_layers, dropout
  - Same learning rate, optimizer, negative sampling ratio
  - Same seeds

Only difference: input is entity ID embedding only. No neighbor triples.
No graph structure of any kind.

This is the controlled baseline for Q1. Any AUPRC delta between this and
TransformerKG is attributable to the KG context, not to model capacity.
"""

import torch
import torch.nn as nn


class TransformerNoKG(nn.Module):
    """
    Q1 no-KG condition. Same backbone as TransformerKG, no triple context.
    """

    def __init__(
        self,
        n_drugs: int,
        n_diseases: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_kg = False  # marker for ablation tracking

        self.drug_emb = nn.Embedding(n_drugs, hidden_dim)
        self.disease_emb = nn.Embedding(n_diseases, hidden_dim)

        self.drug_type_token = nn.Parameter(torch.randn(hidden_dim))
        self.disease_type_token = nn.Parameter(torch.randn(hidden_dim))

        # Identical transformer to TransformerKG
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.score_head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        drug_idx: torch.Tensor,     # [B]
        disease_idx: torch.Tensor,  # [B]
        **kwargs,                   # accepts but ignores neighbor data
    ) -> torch.Tensor:
        drug_base = self.drug_emb(drug_idx) + self.drug_type_token
        disease_base = self.disease_emb(disease_idx) + self.disease_type_token

        drug_seq = drug_base.unsqueeze(1)       # [B, 1, D]
        disease_seq = disease_base.unsqueeze(1) # [B, 1, D]

        drug_enc = self.transformer(drug_seq)[:, 0, :]      # [B, D]
        disease_enc = self.transformer(disease_seq)[:, 0, :]

        pair_rep = torch.cat([drug_enc, disease_enc], dim=-1)
        return self.score_head(pair_rep).squeeze(-1)
