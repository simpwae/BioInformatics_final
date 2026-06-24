# 3. Results

> All numbers in this section link to files in `results/`. Run `python scripts/generate_table.py` to regenerate the table from raw result files.
>
> Models labeled `scaled_reproduction` differ from the published TxGNN in four ways: hidden_dim 512→64, layers 3→2, heads 8→4, learnable embeddings instead of pre-trained node features. Paper numbers go in the `paper_reported` column; our numbers go in `scaled_reproduction`. They are never merged.

## 3.1 Q1 — Does KG Augmentation Improve Over No-KG?

**Experimental design**: same pretrain procedure for both conditions. The no-KG model uses `num_layers=0` (no message passing); the KG model uses 2 HGT layers. Both use the same link prediction pretrain on KG edges to position embeddings.

**Result on standard split (indication AUPRC, seeds [42, 0, 1]):**

| Model | AUPRC ind (mean±std) | AUPRC contra (mean±std) |
|-------|---------------------|------------------------|
| gnn_no_kg | 0.893 ± 0.006 | 0.945 ± 0.002 |
| gnn_kg    | 0.825 ± 0.022 | 0.895 ± 0.026 |

**Result on zero-shot split (indication AUPRC):**

| Model | AUPRC ind (mean±std) | AUPRC contra (mean±std) |
|-------|---------------------|------------------------|
| gnn_no_kg | 0.704 ± 0.027 | 0.725 ± 0.008 |
| gnn_kg    | 0.714 ± 0.026 | 0.831 ± 0.016 |

**Finding**: KG message passing does not improve indication AUPRC on either split in this scaled reproduction. On the standard split, the no-KG model outperforms the KG model. On the zero-shot split, results are within noise for indication; the KG model is better for contraindication (0.831 vs 0.725).

This result is reported as-is, per ground rule 1.2. The hypothesis "KG helps" is not supported by the data on indication prediction. The contraindication zero-shot split shows a KG advantage, which may reflect the larger contraindication class imbalance benefiting from neighborhood regularization.

*Source files: `results/gnn/{standard,zeroshot}/seed_{42,0,1}/gnn_baseline.json`, `gnn_no_kg.json`*

## 3.2 Q2 — Is Two-Phase Training Better?

**Experimental design**: three training regimes compared on the same TxGNN backbone (hidden_dim=64, 2 layers, 4 heads):
- `txgnn_two_phase`: Phase 1 = self-supervised KG link-prediction; Phase 2 = therapeutic task fine-tuning
- `single_stage`: KG link-prediction loss + therapeutic task loss jointly from epoch 0
- `joint_contrastive`: therapeutic task loss + InfoNCE disease-similarity contrastive loss (no KG link-prediction)

**"Better" was defined before running**: primary criterion = higher AUPRC on zero-shot indication at equal or lower wall-clock time.

**Result on zero-shot split (indication AUPRC, seeds [42, 0, 1]):**

| Method | AUPRC ind (mean±std) | AUPRC contra (mean±std) | Wall-clock (s) |
|--------|---------------------|------------------------|----------------|
| txgnn_two_phase | 0.736 ± 0.011 | 0.832 ± 0.010 | 71.6 |
| single_stage    | 0.706 ± 0.033 | 0.833 ± 0.019 | 130.7 |
| joint_contrastive | 0.670 ± 0.038 | 0.854 ± 0.016 | 97.1 |

**Result on standard split (indication AUPRC, seeds [42, 0, 1]):**

| Method | AUPRC ind (mean±std) | AUPRC contra (mean±std) | Wall-clock (s) |
|--------|---------------------|------------------------|----------------|
| txgnn_two_phase | 0.816 ± 0.012 | 0.899 ± 0.010 | 183.8 |
| single_stage    | 0.723 ± 0.005 | 0.863 ± 0.012 | 100.3 |
| joint_contrastive | 0.739 ± 0.019 | 0.862 ± 0.014 | 95.9 |

**Finding**: Two-phase training outperforms both alternatives on zero-shot indication (0.736 vs 0.706 vs 0.670). It also runs faster than single_stage (71.6s vs 130.7s). The joint_contrastive approach has higher zero-shot contraindication AUPRC (0.854) but lower indication (0.670) and shows more variance across seeds (std=0.038). Two-phase training is the better method under the pre-defined criterion.

*Source files: `results/txgnn/`, `results/alt_single_stage/`, `results/alt_joint_contrastive/`*

## 3.3 Q3 — Why Is Zero-Shot Preferred?

**Conceptual answer**: Huang et al. (2024) report that 92% of PrimeKG diseases have no indication edge (PMC11326339). On our downloaded data, 9,388 unique indication pairs cover 17,080 diseases, consistent with that figure. A supervised model has no labels for unlabeled diseases. Zero-shot generalization via disease similarity is the only tractable path for rare disease coverage.

**Empirical**: Degradation curve generated and written to `results/figures/degradation_curve.png`. Data in `results/metrics/degradation_curve_data.json`.

The plot compares:
- GNN baseline (standard split): AUPRC for test diseases that have training treatment edges. Shows high AUPRC (0.8–0.9) for diseases with many training edges, degrading toward the 0-edge regime.
- GNN baseline (zero-shot split): all test diseases have zero training edges. AUPRC ~0.70 (indication).
- TxGNN (zero-shot split): all test diseases have zero training edges. AUPRC ~0.74 (indication).

The degradation from the standard split (diseases with labeled training edges → high AUPRC) to the zero-shot split (zero edges → lower AUPRC) motivates zero-shot prediction: models that can generalize without training labels outperform supervised models in the 0-edge regime. TxGNN's disease-similarity module provides a small but consistent advantage at 0 training edges.

*Source: `results/figures/degradation_curve.png`, `results/metrics/degradation_curve_data.json`, seed=42.*

## 3.4 Q4 — Case Studies

See Section 4 for full case study content.

**Case A**: Familial Hypertrophic Cardiomyopathy — rare, n_pos=1 in PrimeKG zeroshot test. The model's top-20 indication predictions do not include Propranolol (the one known indication). AUPRC=0.025.

**Case B**: Staphylococcus Aureus Infection — 45 approved therapies in PrimeKG zeroshot test. Benzylpenicillin appears at rank 18. Mupirocin (rank 3) and Doxycycline (rank 5) are clinically plausible. AUPRC=0.088.

*Source files: `results/predictions/case_study_caseA_txgnn.csv`, `results/predictions/case_study_caseB_txgnn.csv`*

## 3.5 Q5 — GNN vs TxGNN Comparison Table

*Full table auto-generated by `scripts/generate_table.py` from `results/`. All metrics use random-negative AUPRC (1:5 ratio). Means and standard deviations across seeds [42, 0, 1].*

### Standard Split

| Model | AUPRC ind (scaled_reproduction) | AUPRC ind (paper_reported) | AUPRC contra (scaled_reproduction) | Wall-clock (s) |
|-------|---------------------------------|----------------------------|-------------------------------------|----------------|
| gnn_no_kg | 0.893 ± 0.006 | — | 0.945 ± 0.002 | 0.7 |
| gnn_kg (HAN equiv.) | 0.825 ± 0.022 | 0.873 ¹ | 0.895 ± 0.026 | 145.5 |
| txgnn_two_phase | 0.816 ± 0.012 | **0.913** ¹ | 0.899 ± 0.010 | 183.8 |
| single_stage | 0.723 ± 0.005 | — | 0.863 ± 0.012 | 100.3 |
| joint_contrastive | 0.739 ± 0.019 | — | 0.862 ± 0.014 | 95.9 |

¹ Huang et al. (2024), PMC11326339 (preprint); HAN was the best GNN-family baseline in the paper. Peer-reviewed final (PMC11645266) not yet retrieved — numbers may differ slightly.

### Zero-Shot Split

| Model | AUPRC ind (scaled_reproduction) | AUPRC ind (paper_reported) | AUPRC contra (scaled_reproduction) | Wall-clock (s) |
|-------|---------------------------------|----------------------------|-------------------------------------|----------------|
| gnn_no_kg | 0.704 ± 0.027 | — | 0.725 ± 0.008 | 0.2 |
| gnn_kg | 0.714 ± 0.026 | — | 0.831 ± 0.016 | 71.3 |
| txgnn_two_phase | 0.736 ± 0.011 | [see Suppl. S1/S2] ² | 0.832 ± 0.010 | 71.6 |
| single_stage | 0.706 ± 0.033 | — | 0.833 ± 0.019 | 130.7 |
| joint_contrastive | 0.670 ± 0.038 | — | 0.854 ± 0.016 | 97.1 |

² Absolute zero-shot AUPRC values are in Supplementary Tables 1–2 (MOESM1 ESM) only — confirmed absent from the main text of both the preprint (PMC11326339) and the peer-reviewed article (PMC11645266). Both versions give the same relative gains only: +19.0% indication / +23.9% contraindication over next-best, random zero-shot split (Fig 2d). The abstract headline (+49.2% / +35.1%) appears in both versions but is not tied to any specific split in the main text; it is not used as a per-split figure. To fill this cell with absolute values, download the SI file (MOESM1 ESM) from PMC11645266 and extract the relevant rows from Suppl. Table 1.

*Source (scaled_reproduction): `results/metrics/comparison_table.csv`. Source (paper_reported): Huang et al. (2024), "A foundation model for clinician-centered drug repurposing," Nature Medicine, PMC11326339.*

**Summary of Q5 findings**:
- On the standard split, gnn_no_kg achieves the highest indication AUPRC (0.893). Two-phase TxGNN (0.816) is 7.7 points below the no-KG baseline.
- On the zero-shot split, TxGNN two-phase leads on indication (0.736), followed by gnn_kg (0.714), single_stage (0.706), gnn_no_kg (0.704), and joint_contrastive (0.670).
- TxGNN's advantage on zero-shot is 3.2 points over gnn_no_kg — meaningful but modest in this scaled reproduction.
- Wall-clock: gnn_no_kg is fastest (0.7–0.2s). TxGNN and gnn_kg are comparable (~70s). single_stage is slowest (130s).

## 3.6 Q6 — Is Attention Optional?

**Decision rule (set before running)**:
- delta = AUPRC(attn=ON) − AUPRC(attn=OFF) on zero-shot indication
- delta < 0.02: attention is optional
- delta ≥ 0.02: attention matters
- Extended rule added before seeing results: if delta < -0.02, attention is actively detrimental

**Full ablation table (zero-shot split, seeds [42, 0, 1]):**

| Variant | AUPRC ind (mean±std) | AUPRC contra (mean±std) | Wall-clock (s) |
|---------|---------------------|------------------------|----------------|
| txgnn (attn=ON, sim=ON) | 0.736 ± 0.011 | 0.832 ± 0.010 | 71.6 |
| txgnn_no_attn (attn=OFF, sim=ON) | 0.772 ± 0.013 | 0.787 ± 0.008 | 3.2 |
| txgnn_no_sim (attn=ON, sim=OFF) | 0.726 ± 0.007 | 0.824 ± 0.008 | — |
| txgnn_no_both (attn=OFF, sim=OFF) | 0.762 ± 0.010 | 0.780 ± 0.009 | — |

**Q6 Decision**: delta = 0.736 − 0.772 = **-0.036**. Delta < -0.02, therefore attention is **DETRIMENTAL** in this scaled reproduction. Removing HGT attention and using mean aggregation instead improves zero-shot indication AUPRC by 3.6 points.

**Interpretation**: The hypothesis (zero-shot advantage comes from disease-similarity, not attention) is partially supported:
- Removing attention improves performance — attention is not load-bearing, it hurts.
- Removing similarity (txgnn_no_sim, 0.726) hurts slightly vs. the full model (0.736), confirming the similarity module helps.
- Removing both components (txgnn_no_both, 0.762) is better than the full model but worse than removing attention alone (0.772), suggesting attention and similarity interact.
- The disease-similarity module is the more beneficial of the two components in this scaled setup.
- The HGT attention mechanism may overfit with only 64-dimensional embeddings and limited training data. The published 512-dim model may behave differently.

*Source: `results/ablations/matrix.json`, `results/metrics/q6_ablation_table.csv`*
