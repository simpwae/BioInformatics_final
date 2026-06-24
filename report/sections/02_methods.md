# 2. Methods

## 2.1 Data and Splits

PrimeKG (Chandak et al., 2023) — downloaded from Harvard Dataverse (dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM). 8,100,498 total edges. License: CC0.

**Standard split**: train/val/test on (drug, disease) pairs, diseases appear in all sets.
**Zero-shot split**: held-out diseases have zero treatment edges in train. 641 held-out diseases (all with zero approved therapies in training).

Leakage check: `scripts/run_leakage_check.py` verified 0 leaking diseases for seeds [42, 0, 1]. Results in `results/metrics/leakage_check_seed{n}.json`.

Edges excluded from GNN message passing: `anatomy_protein_present` (3.03M) and `drug_drug` (2.67M) — excluded due to VRAM constraint. Remaining: ~2.4M of 8.1M edges. Documented as scaled_reproduction deviation.

## 2.2 Models

### GNN Baseline (Q1 KG condition)
Two-layer HGT encoder. Node features: `nn.Embedding` per type (Xavier uniform). Hidden dim: 64. Scoring: dot product. Phase 1: link prediction pretrain (30 epochs). Phase 2: fine-tune on therapeutic edges (up to 100 epochs, early stopping patience=10).

### GNN No-KG Baseline (Q1 no-KG condition)
Same as GNN baseline but `num_layers=0`. Encode returns raw embeddings with no message passing. Both conditions share the same Phase 1 pretrain (link prediction using KG edges to position embeddings). The no-KG label refers to absence of message passing, not absence of KG-informed pretraining.

### TxGNN (scaled reproduction)
Same HGT encoder as GNN baseline. Added: `DiseaseSimilarityModule` (cosine similarity projection, k=5 nearest support diseases). Two-phase training: Phase 1 = link prediction pretrain; Phase 2 = therapeutic task + triplet-style metric-learning loss (`sim_loss_weight=0.3`). `use_attention=True, use_similarity=True`.

Deviations from paper: hidden_dim 512→64, num_layers 3→2, num_heads 8→4, node features are learnable embeddings (paper uses pre-trained features — inaccessible).

### Alternatives (Q2)
- **SingleStageModel**: KG link prediction + therapeutic task jointly from epoch 1. No Phase 1/Phase 2 split.
- **JointContrastiveModel**: InfoNCE disease similarity + therapeutic task jointly.

### Ablations (Q6)
- **TxGNN attn=ON**: full model (HGT attention).
- **TxGNN attn=OFF**: SAGEConv with mean aggregation instead of HGT attention.

## 2.3 Evaluation

**Primary metric**: AUPRC (Area Under Precision-Recall Curve).
**Secondary metric**: AUROC.
**Evaluation protocol**: random negative sampling (1:5 positive:negative ratio) for all flat evaluations. Per-disease full-ranking evaluation for TxGNN zero-shot results.
**Seeds**: [42, 0, 1]. Report mean ± std.
**Table generation**: `scripts/generate_table.py` reads result JSON files. Never type numbers manually.

## 2.4 Leakage Check

Zero-shot results are only valid after `scripts/run_leakage_check.py` passes. Output: `results/metrics/leakage_check_seed{n}.json`. Status: PASS for all seeds (verified before running any zero-shot experiment).
