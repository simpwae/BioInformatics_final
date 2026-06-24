# Data Inventory for Web Interface

Generated 2026-06-24. Documents all result files consumed by `scripts/build-data.mjs`.

---

## CSV Files

### `results/metrics/comparison_table.csv`
Serves: Q1, Q2, Q5
Columns: model, split, n_seeds_run, reproduction_type, auprc_ind, auprc_ind_std, auroc_ind, auroc_ind_std, auprc_contra, auprc_contra_std, auroc_contra, auroc_contra_std, wall_s
Rows: gnn_no_kg, gnn_kg, txgnn_two_phase, single_stage, joint_contrastive, txgnn_attn_on, txgnn_attn_off (x standard + zeroshot)
Excluded rows: transformer_kg, transformer_nokg (values = "[NOT YET RUN]")

### `results/metrics/q6_ablation_table.csv`
Serves: Q6
Columns: variant, split, auprc_ind_mean, auprc_ind_std, n_seeds

---

## JSON Files

### `results/ablations/matrix.json`
Serves: Q6
Keys: q6_decision{status, delta_attn_on_minus_off, threshold, attention_optional, conclusion}, matrix[...]

### `results/metrics/degradation_curve_data.json`
Serves: Q3
Array of {model, disease_id, relation, n_train_edges, auprc}

---

## Prediction CSVs

### `results/predictions/case_study_caseA_txgnn.csv`
Disease: 24573 (Familial Hypertrophic Cardiomyopathy), indication + contraindication top-20

### `results/predictions/case_study_caseB_txgnn.csv`
Disease: 5545 (Staphylococcus Aureus Infection), indication top-20

### `results/predictions/case_study_caseA_paths_txgnn.csv`
KG paths for Case A drugs

### `results/predictions/case_study_caseB_paths_txgnn.csv`
KG paths for Case B drugs

---

## Report Sections

`report/sections/01_introduction.md` through `05_discussion.md`

---

## Key paper_reported Values (from `lit/notes.md`, Huang et al. 2024 Nat Med, Suppl. S1/S2)

Standard split:
- TxGNN indication AUPRC: 0.91 +/- 0.02
- HAN indication AUPRC: 0.87 +/- 0.18
- TxGNN contraindication AUPRC: 0.82 +/- 0.01
- HAN contraindication AUPRC: 0.84 +/- 0.00

Zero-shot split:
- TxGNN indication AUPRC: 0.90 +/- 0.02
- TxGNN contraindication AUPRC: 0.80 +/- 0.01
- Relative gains: +19.0% ind / +23.9% contra vs next-best baseline
