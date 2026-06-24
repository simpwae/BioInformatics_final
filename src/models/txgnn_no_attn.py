"""
TxGNN ablation: attention OFF, similarity module ON.

This is Q6 condition B. Instantiates TxGNN with use_attention=False.
Mean aggregation replaces HGT attention.

If AUPRC delta (attn_on - attn_off) < 0.02 on zero-shot split -> attention is optional.
If delta >= 0.02 -> attention matters; report that.
Decision rule set in CLAUDE.md before running.
"""

from src.models.txgnn import TxGNN


def build_txgnn_no_attn(metadata: tuple, node_counts: dict = None, hidden_dim: int = 64, **kwargs) -> TxGNN:
    return TxGNN(
        metadata=metadata,
        node_counts=node_counts,
        hidden_dim=hidden_dim,
        use_attention=False,
        use_similarity=True,
        **kwargs,
    )


def build_txgnn_no_sim(metadata: tuple, node_counts: dict = None, hidden_dim: int = 64, **kwargs) -> TxGNN:
    """Ablation: attention ON, similarity module OFF."""
    return TxGNN(
        metadata=metadata,
        node_counts=node_counts,
        hidden_dim=hidden_dim,
        use_attention=True,
        use_similarity=False,
        **kwargs,
    )


def build_txgnn_no_both(metadata: tuple, node_counts: dict = None, hidden_dim: int = 64, **kwargs) -> TxGNN:
    """Both OFF. Baseline equivalent to plain GNN with no special modules."""
    return TxGNN(
        metadata=metadata,
        node_counts=node_counts,
        hidden_dim=hidden_dim,
        use_attention=False,
        use_similarity=False,
        **kwargs,
    )
