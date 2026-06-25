# 4. Case Studies (Q4)

> All predicted drug scores and KG paths come from model output files in `results/predictions/`.
> Numbers are not typed by hand.

## 4.0 Relationship to Paper's Case Studies

Huang et al. (2024) include their own case studies in the Nature Medicine paper. Their case studies use specific diseases analyzed with the **full published TxGNN** (hidden_dim=512, 3 layers, pre-trained node features). This reproduction **does not replicate those specific case studies** because:

1. The full-scale model could not be run on 8 GB VRAM — all results use a scaled_reproduction (hidden_dim=64).
2. The paper's specific disease selections may not appear in the zero-shot test split of our downloaded PrimeKG version.
3. The paper's pre-trained node features are not publicly accessible; our model uses learnable embeddings.

This section runs its own case studies on the scaled_reproduction model, chosen from diseases that **actually appear in our zero-shot test split** with sufficient positive labels to evaluate. These are not claimed to match the paper's narrative — they are original case studies on the scaled model.

**What the paper's case studies show vs. what ours show:**

| Aspect | Paper (Huang et al. 2024) | This reproduction |
|--------|--------------------------|-------------------|
| Model | Full TxGNN, 512-dim, pre-trained features | Scaled TxGNN, 64-dim, learnable embeddings |
| Disease selection | Paper's own choice | Constrained to zero-shot test split presence |
| Case A type | Rare disease, paper's selection | Familial Hypertrophic Cardiomyopathy (n_pos=1) |
| Case B type | Well-studied disease, paper's selection | Staphylococcus Aureus Infection (n_pos=45) |
| Claim | Published model succeeds on rare diseases | Scaled model fails on n_pos=1; partial success on n_pos=45 |

## 4.0b Disease Selection (This Reproduction)

Selected from the actual per_disease_results in `results/txgnn/zeroshot/seed_42/txgnn.json`. Both diseases were chosen before examining the top-K drug predictions (the disease was fixed; only AUPRC and n_pos were checked at selection time).

**Case A (rare, zero-shot):** Familial Hypertrophic Cardiomyopathy (id=24573)
- Rationale: inherited cardiac muscle disease, n_pos=1 in the zeroshot test split. Only one approved therapy exists in PrimeKG for this disease.
- Changed from original selection (Hutchinson-Gilford Progeria Syndrome) after discovering Progeria has zero therapeutic edges in all PrimeKG splits. Change logged in CONTEXT.md.

**Case B (well-studied, zero-shot):** Staphylococcus Aureus Infection (id=5545)
- Rationale: common bacterial infection, n_pos=45 approved therapies in PrimeKG zeroshot test split. Corresponds to the "well-studied disease with multiple approved therapies" requirement from CLAUDE.md.
- Both cases drawn from the zeroshot split because only zeroshot results contain per-disease `top_k_drugs` (standard split was not re-evaluated; no checkpoint saved).
- Changed from original selection (Type 2 Diabetes) after discovering id=5148 does not appear in the zeroshot/seed_42 test split.

---

## 4.1 Model and Output Files

- Model: TxGNN (scaled reproduction, `scaled_reproduction`), zeroshot split, seed=42
- Prediction file (Case A): `results/predictions/case_study_caseA_txgnn.csv`
- Prediction file (Case B): `results/predictions/case_study_caseB_txgnn.csv`
- KG paths (Case A): `results/predictions/case_study_caseA_paths_txgnn.csv`
- KG paths (Case B): `results/predictions/case_study_caseB_paths_txgnn.csv`

To regenerate: `python src/case_studies/case_study_runner.py`

---

## 4.2 Case A: Familial Hypertrophic Cardiomyopathy (id=24573, n_pos=1)

**Background**: Familial hypertrophic cardiomyopathy (FHC) is a hereditary condition causing pathological thickening of the heart muscle. It is caused by mutations in sarcomere genes (MYH7, MYBPC3, others). The only indication edge for this disease in PrimeKG is Propranolol (DB00571), a non-selective beta-blocker used to reduce outflow obstruction.

**Top-10 indication predictions (from `case_study_caseA_txgnn.csv`, seed=42):**

| Rank | Drug | DrugBank ID | Score | Positive |
|------|------|-------------|-------|---------|
| 1 | Lutetium Lu 177 dotatate | DB13985 | -1.038 | No |
| 2 | Lactose | DB04465 | -1.206 | No |
| 3 | Emapalumab | DB14724 | -1.226 | No |
| 4 | Cysteine | DB00151 | -1.234 | No |
| 5 | Sulfapyridine | DB00891 | -1.251 | No |
| 6 | Etoposide | DB00773 | -1.263 | No |
| 7 | Oleandomycin | DB11442 | -1.265 | No |
| 8 | Etretinate | DB00926 | -1.268 | No |
| 9 | Quinacrine | DB01103 | -1.283 | No |
| 10 | Umifenovir | DB13609 | -1.293 | No |

*All scores negative. None of the top-20 predictions are positive. Propranolol (the one known indication) does not appear in the top-20.*

**Per-disease AUPRC (indication, seed=42)**: 0.025 (from result file). The model fails on this disease — a very low score that confirms poor ranking of the single positive drug.

**KG paths for known drugs (from `case_study_caseA_paths_txgnn.csv`):**

The three drugs traced (Milrinone DB00235, Amrinone DB01427, Dipyridamole DB00975) are all **contraindications** for FHC (not positive indications). Their 2-hop paths share a pattern:

```
Drug → [contraindication] → hypertrophic cardiomyopathy → [disease_disease] → familial hypertrophic cardiomyopathy (id=24573)
Drug → [drug_drug] → Propranolol → [indication] → familial hypertrophic cardiomyopathy
```

The model reaches FHC primarily through the broader "hypertrophic cardiomyopathy" disease node (disease similarity), not through drug-specific paths. This confirms the model uses disease-disease similarity propagation for rare diseases with few direct edges.

**Clinical note**: The top predictions (Lutetium Lu 177 dotatate — a cancer radiopharmaceutical; Lactose — an excipient) are clinically implausible for FHC. The model does not correctly rank Propranolol in the top 20. This is consistent with the AUPRC of 0.025 and with the model's limited capacity (64-dim embeddings) to generalize to diseases with only one known drug in the KG.

---

## 4.3 Case B: Staphylococcus Aureus Infection (id=5545, n_pos=45)

**Background**: Staphylococcus aureus is a common gram-positive bacterial pathogen responsible for skin infections, pneumonia, endocarditis, and sepsis. Multiple antibiotic classes are indicated, including beta-lactams, tetracyclines, macrolides, and topical agents. MRSA strains require mupirocin, daptomycin, or vancomycin.

**Top-10 indication predictions (from `case_study_caseB_txgnn.csv`, seed=42):**

| Rank | Drug | DrugBank ID | Score | Positive |
|------|------|-------------|-------|---------|
| 1 | Sulfapyridine | DB00891 | 4.783 | No |
| 2 | Etoposide | DB00773 | 4.708 | No |
| 3 | Mupirocin | DB00410 | 4.686 | No |
| 4 | Oleandomycin | DB11442 | 4.664 | No |
| 5 | Doxycycline | DB00254 | 4.659 | No |
| 6 | Carboplatin | DB00958 | 4.590 | No |
| 7 | Emapalumab | DB14724 | 4.558 | No |
| 8 | Bleomycin | DB00290 | 4.542 | No |
| 9 | Phenylbutyric acid | DB06819 | 4.537 | No |
| 10 | Minocycline | DB01017 | 4.524 | No |
| 18 | Benzylpenicillin | DB01053 | 4.366 | **Yes** |

*First positive drug appears at rank 18: Benzylpenicillin (Penicillin G). Three of the top-10 predictions are cancer drugs (Etoposide, Carboplatin, Bleomycin), which are not indicated for bacterial infections.*

**Per-disease AUPRC (indication, seed=42)**: 0.088 (from result file).

**Clinically plausible predictions in the top-20:**
- Rank 3: Mupirocin — topical antibiotic, first-line for MRSA decolonization and skin infections
- Rank 5: Doxycycline — broad-spectrum antibiotic, active against S. aureus
- Rank 10: Minocycline — tetracycline-class antibiotic with S. aureus activity
- Rank 4: Oleandomycin — macrolide antibiotic (old)
- Rank 18: Benzylpenicillin — first-line for methicillin-susceptible S. aureus (MSSA)

**Clinically implausible predictions:**
- Ranks 2, 6, 8 (Etoposide, Carboplatin, Bleomycin): antineoplastic drugs; no role in bacterial infection

**KG paths for known drugs (from `case_study_caseB_paths_txgnn.csv`):**

The three approved drugs traced (Cefprozil DB01150, Cefdinir DB00535, Tazobactam DB01606) share a common path pattern:

```
Drug → [indication] → staphylococcal infection → [disease_disease] → staphylococcus aureus infection (id=5545)
Drug → [drug_drug] → Ceftazidime → [indication] → staphylococcus aureus infection
```

The model's primary reasoning path goes through a related disease node ("staphylococcal infection") and then connects via disease-disease similarity to the specific target disease. This is the same mechanism seen in Case A — disease similarity is the principal reasoning path.

**Discrepancy with known clinical evidence**: The three implausible cancer drug predictions (Etoposide, Carboplatin, Bleomycin at ranks 2, 6, 8) likely result from these drugs sharing KG neighbors (perhaps immune-related nodes) with the anti-infective drugs. In a 64-dim embedding space with no pre-trained features, the model cannot distinguish mechanism of action at this resolution.

---

## 4.4 Interpretation Notes

- Both case studies use the TxGNN scaled reproduction (hidden_dim=64, no pre-trained node features). Rankings should not be interpreted as clinical recommendations.
- The disease-disease similarity pathway dominates both case studies, confirming the Q6 finding that the similarity module is the load-bearing component.
- For Case A (rare disease, n_pos=1), the model fails to rank the approved drug in the top 20. The zero-shot claim "generalization to rare diseases" is not confirmed for n_pos=1 diseases in this scaled reproduction.
- For Case B (well-studied, n_pos=45), several clinically plausible antibiotics appear in the top-20 (Mupirocin rank 3, Doxycycline rank 5, Benzylpenicillin rank 18). But three cancer drugs appear in ranks 2, 6, 8 — a signal of noise in the embeddings.
- These observations are specific to seed=42, zeroshot split, scaled_reproduction model. They are not claims about the published TxGNN.
