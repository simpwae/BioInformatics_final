# 5. Discussion

## 5.1 KG Augmentation Does Not Consistently Help

On PrimeKG, standard split, seeds [42, 0, 1], the no-KG embedding model (pretrained on KG edges, no message passing) outperforms the 2-layer HGT on indication AUPRC (0.893 ± 0.006 vs. 0.825 ± 0.022). On the zero-shot split, the two are within noise for indication.

This is not a universal claim. It is a finding specific to:
- PrimeKG, 2026-06-24 version
- Scaled reproduction (hidden_dim=64, 2 layers, edges filtered to ~2.4M)
- Random-negative evaluation (1:5 ratio)
- Seeds [42, 0, 1]

Several explanations are consistent with this finding: (a) HGT with 2 layers over non-therapeutic edges adds noise to drug/disease representations; (b) the random-negative evaluation inflates AUPRC for both conditions but may inflate the no-KG condition more (it memorizes train patterns well); (c) the publication's 512-dim embeddings likely have more representational capacity to use KG structure effectively.

This finding should not be generalized beyond the specific setup described.

## 5.2 Zero-Shot vs. Standard Coverage

76.5% of PrimeKG diseases (13,075 of 17,080) have no approved therapy in the training data. A standard supervised model has no labels for any of these. This is the motivating constraint for zero-shot prediction — it is not a design preference but a necessity given the data.

## 5.3 Scaled Reproduction Limitations

Four deviations from the published TxGNN apply to all results here:
1. hidden_dim: 512 → 64 (VRAM constraint)
2. num_layers: 3 → 2 (VRAM constraint)
3. num_heads: 8 → 4 (VRAM constraint)
4. Node features: pre-trained features → learnable embeddings (published features inaccessible)

These deviations compound. A model with 64-dim embeddings rather than 512 has 8× less representational capacity. Results should not be compared to the paper's reported numbers. The `scaled_reproduction` label marks all our outputs.

The paper's numbers (Huang et al., 2024, Nature Medicine) go in the `paper_reported` column of the comparison table. Our numbers go in `scaled_reproduction`. They are never merged.

## 5.4 Evaluation Protocol Note

GNN baseline results use random-negative AUPRC (1:5 ratio). TxGNN results include both random-negative (field: `indication_flat.auprc`) and per-disease full-ranking AUPRC (`indication.auprc`). The Q5 comparison table uses random-negative metrics for cross-model consistency. The random-negative metric is inflated relative to full ranking but is consistent across models.
