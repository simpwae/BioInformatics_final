"""
Evaluation metrics for drug-disease link prediction.
All functions take numpy arrays or tensors.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve."""
    return float(average_precision_score(y_true, y_score))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    return float(roc_auc_score(y_true, y_score))


def hits_at_k(
    scores: np.ndarray,
    pos_mask: np.ndarray,
    k: int = 10,
) -> float:
    """
    Hits@K: fraction of positive (drug, disease) pairs that appear in
    the top-K ranked candidates for their disease.

    scores:   [N_drugs] scores for a single disease
    pos_mask: [N_drugs] boolean, True = positive drug
    """
    top_k_idx = np.argsort(scores)[::-1][:k]
    hits = pos_mask[top_k_idx].sum()
    n_pos = pos_mask.sum()
    if n_pos == 0:
        return float("nan")
    return float(hits / n_pos)


def evaluate_flat(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task: str,
) -> dict:
    """
    Flat evaluation (all pairs pooled). Returns dict matching results JSON schema.
    y_true: [N] binary labels
    y_score: [N] model scores (higher = more likely positive)
    task: "indication" or "contraindication"
    """
    result = {
        "task": task,
        "n_pairs": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "auprc": auprc(y_true, y_score),
        "auroc": auroc(y_true, y_score),
        "prevalence": float(y_true.mean()),
    }
    return result
