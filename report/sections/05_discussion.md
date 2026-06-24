# 5. Discussion

## 5.1 KG Augmentation Does Not Consistently Help (Q1)

On PrimeKG, standard split, seeds [42, 0, 1], the no-KG embedding model (pretrained on KG edges, no message passing) outperforms the 2-layer HGT on indication AUPRC (0.893 ± 0.006 vs. 0.825 ± 0.022). On the zero-shot split, the two are within noise for indication (0.704 vs. 0.714), though the KG model shows a clear advantage for contraindication (0.831 vs. 0.725).

This is not a universal claim. It is a finding specific to:
- PrimeKG, evaluated 2026-06-24
- Scaled reproduction (hidden_dim=64, 2 layers, edges filtered to ~2.4M)
- Random-negative evaluation (1:5 ratio)
- Seeds [42, 0, 1]

Several explanations are consistent with this finding: (a) HGT with 2 layers over non-therapeutic edges adds noise to drug/disease representations; (b) with only 64-dim embeddings, the model may lack the capacity to usefully encode KG neighborhood information; (c) the published 512-dim model likely has enough capacity to use KG structure effectively.

This finding should not be generalized beyond the specific setup described.

## 5.2 Two-Phase Training Remains Best for Zero-Shot (Q2)

Neither alternative to TxGNN's two-phase training beats it on zero-shot indication AUPRC under the pre-defined criterion. Two-phase training (0.736 ± 0.011) outperforms single-stage joint training (0.706 ± 0.033) and joint contrastive (0.670 ± 0.038), and also runs faster than both alternatives (71.6s vs 130.7s vs 97.1s).

The pre-training phase (KG link prediction) appears to position embeddings in a space where the subsequent fine-tuning on therapeutic tasks generalizes better to unseen diseases. Collapsing the two phases loses this benefit.

The joint contrastive approach shows higher zero-shot contraindication AUPRC (0.854) but lower and more variable indication AUPRC. This trade-off may be worth investigating further, but under the pre-defined criterion (indication AUPRC), it does not beat two-phase training.

## 5.3 Zero-Shot vs. Standard Coverage (Q3)

76.5% of PrimeKG diseases (13,075 of 17,080) have no approved therapy in the training data. A standard supervised model has no labels for any of these. This is the motivating constraint for zero-shot prediction — it is not a design preference but a necessity given the data.

The empirical degradation curve (`results/figures/degradation_curve.png`) confirms this: GNN baseline AUPRC drops from 0.8–0.9 (diseases with many training edges) toward the zero-edge regime, while the zero-shot TxGNN model maintains 0.74 AUPRC for all test diseases regardless of training edge count.

## 5.4 Case Study Observations (Q4)

For Familial Hypertrophic Cardiomyopathy (n_pos=1, rare), the model fails to rank Propranolol in the top 20. The primary reasoning path goes through disease-disease similarity (hypertrophic cardiomyopathy → FHC). This scaled reproduction does not confirm the zero-shot rare-disease generalization claim for n_pos=1 diseases.

For Staphylococcus Aureus Infection (n_pos=45, well-studied), several clinically plausible antibiotics appear in the top-20 (Mupirocin rank 3, Doxycycline rank 5, Benzylpenicillin rank 18). Three antineoplastic drugs also appear in the top-10, indicating residual noise in the 64-dim embedding space. The reasoning path again relies on disease-disease similarity.

## 5.5 Attention Is Detrimental in This Scaled Reproduction (Q6)

The Q6 ablation shows that removing HGT attention (mean aggregation instead) improves zero-shot indication AUPRC from 0.736 to 0.772 — a delta of -0.036, exceeding the pre-defined threshold of 0.02 in the harmful direction.

The hypothesis that attention is optional is not confirmed in a simple sense: it is stronger than "optional" — attention actively hurts in this setup. The disease-similarity module (not the attention mechanism) is the component that matters: removing similarity (txgnn_no_sim) reduces AUPRC from 0.736 to 0.726.

A likely explanation: HGT attention has O(num_heads × hidden_dim²) parameters per layer. With only 64 dimensions and limited training data, the attention weights overfit. Mean aggregation, having no learned parameters, provides more stable aggregation. The published 512-dim model may behave differently given its 8× larger embedding capacity.

This finding is specific to: scaled reproduction, RTX 4060 8 GB VRAM, hidden_dim=64, 2 layers, PrimeKG, seeds [42, 0, 1]. It is not a claim about HGT attention in general or about the published TxGNN.

## 5.6 Scaled Reproduction Limitations

Four deviations from the published TxGNN apply to all results here:
1. hidden_dim: 512 → 64 (VRAM constraint)
2. num_layers: 3 → 2 (VRAM constraint)
3. num_heads: 8 → 4 (VRAM constraint)
4. Node features: pre-trained features → learnable embeddings (published features inaccessible)

These deviations compound. A model with 64-dim embeddings rather than 512 has 8× less representational capacity. Results should not be compared to the paper's reported numbers. The `scaled_reproduction` label marks all our outputs.

The paper's numbers (Huang et al., 2024, Nature Medicine) go in the `paper_reported` column of the comparison table. Our numbers go in `scaled_reproduction`. They are never merged.

## 5.7 Evaluation Protocol Note

GNN baseline results and TxGNN results both use random-negative AUPRC (1:5 ratio) for the Q5 comparison table, ensuring cross-model consistency. TxGNN result files also contain per-disease full-ranking AUPRC (field: `indication.auprc`) which is much lower in absolute value (~0.009–0.014) due to the 1:7957 class imbalance when all drugs are candidates. The per-disease metric is used internally for the degradation curve (Q3) and case study AUPRC values; the random-negative metric is used in all cross-model comparisons (Q1, Q2, Q5, Q6).
