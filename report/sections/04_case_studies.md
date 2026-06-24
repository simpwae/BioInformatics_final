# 4. Case Studies (Q4)

> All predicted drug scores and KG paths come from model output files in `results/predictions/`.
> Numbers are not typed by hand. If a file does not exist, it is marked [NOT YET RUN].

## 4.0 Disease Selection

Selected **before** running any models, per CLAUDE.md Section 2 (Q4):

**Case A (zero-shot):** Hutchinson-Gilford Progeria Syndrome
- Rationale: rare genetic disease; expected to have zero approved drug-disease treatment edges in PrimeKG training data.
- Selected before seeing any model predictions.

**Case B (standard):** Type 2 diabetes mellitus
- Rationale: well-studied metabolic disease with multiple approved drugs (metformin, GLP-1 agonists, insulin, SGLT2 inhibitors, etc.). Expected to have multiple indication edges in training.
- Selected before seeing any model predictions.

---

## 4.1 Model and Output Files

- Model: TxGNN (scaled reproduction), zero-shot split, seed=42
- Prediction file (Case A): `results/predictions/case_study_caseA_txgnn.csv`
- Prediction file (Case B): `results/predictions/case_study_caseB_txgnn.csv`
- KG paths (Case A): `results/predictions/case_study_caseA_paths_txgnn.csv`
- KG paths (Case B): `results/predictions/case_study_caseB_paths_txgnn.csv`

To regenerate: `python src/case_studies/case_study_runner.py`

---

## 4.2 Case A: Hutchinson-Gilford Progeria Syndrome (Zero-Shot)

[NOT YET RUN] — prediction file pending TxGNN zeroshot/seed=42 re-run with top_k_drugs populated.

Once available, this section will report:
1. Top-10 predicted indication drugs (rank, drug name, score, is_positive)
2. Top-10 predicted contraindication drugs
3. Two drugs traced through their 2-hop KG paths to the disease
4. Note on any discrepancy with known clinical evidence

---

## 4.3 Case B: Type 2 Diabetes Mellitus (Standard)

[NOT YET RUN] — prediction file pending.

Once available, this section will report:
1. Top-10 predicted indication drugs (rank, drug name, score, is_positive)
2. Top-10 predicted contraindication drugs
3. Two drugs traced through their 2-hop KG paths to the disease
4. Note on any discrepancy with known clinical evidence (e.g., whether metformin appears in top-10)

---

## 4.4 Interpretation Notes

These case studies are illustrative. Predictions come from a scaled reproduction (hidden_dim=64, no pre-trained node features), not the full TxGNN from Huang et al. (2024). Drug rankings from this model should not be interpreted as clinical recommendations.

The purpose of these case studies is:
1. To verify the model produces plausible rankings (approved drugs rank higher than random)
2. To trace the KG pathways that ground the prediction in biological structure
3. To identify unexpected predictions and flag them for investigation, not discard them
