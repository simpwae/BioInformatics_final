# CLAUDE.md — Project Instructions

> Read this file completely before touching any code, writing any number, or making any claim.
> These are hard constraints. Not suggestions.

---

## 0. Hardware

**GPU:** NVIDIA GeForce RTX 4060, 8 GB VRAM (confirmed via `nvidia-smi`).
**Always use CUDA.** Every script and notebook must call `torch.cuda.is_available()` at startup
and abort with a clear message if CUDA is not found. Never silently fall back to CPU.
Set `CUDA_VISIBLE_DEVICES=0` in every run config. Do not auto-select devices — be explicit.

```python
import torch
assert torch.cuda.is_available(), "CUDA not available — check GPU setup before running."
device = torch.device("cuda:0")
```

Because VRAM is 8 GB, models must be sized to fit. TxGNN is a scaled reproduction.
**Label every result file that uses a scaled model as `scaled_reproduction`** in its filename
and in the header of the results CSV/JSON.

---

## 1. Ground Rules (Non-Negotiable)

### 1.1 Every number must trace to a file
Every metric in the report, every number on the website, must link to a file in `results/metrics/`.
Format: `results/metrics/{model}_{split}_{seed}.json`.
If a number cannot be looked up in a results file or backed by a cited primary paper, it does not appear. Ever.

### 1.2 Failed / null results are results
If KG augmentation does not beat the no-KG baseline, the report says so plainly.
Do not retune hyperparameters until the hypothesis "wins." Run the experiment as designed, record the outcome, report it.

### 1.3 No universal claims
Write:
> "On PrimeKG, under the zero-shot split, with seeds [42, 0, 1], ..."

Do NOT write:
> "Transformers are better with knowledge graphs."
> "KG augmentation consistently improves drug repurposing."

Scope every claim to the exact dataset, split, seeds, and model variant used.

### 1.4 Separate reproduction from original work
TxGNN is reproduced at reduced scale due to the 8 GB VRAM constraint.
Label this everywhere: filenames, result headers, report sections, table captions.
The published numbers from Huang et al. (2024) go in a separate column labeled `paper_reported`.
Our numbers go in a column labeled `scaled_reproduction`. Never merge them.

### 1.5 Leakage is the cardinal sin
The zero-shot claim is only valid if held-out diseases had **zero treatment edges during training**.
`scripts/run_leakage_check.py` must pass before any zero-shot results are accepted.
If leakage is found, stop. Fix the split. Re-run everything. Do not present leaked results.
The leakage check output goes to `results/metrics/leakage_check.json`.

### 1.6 Cite primary sources only
Acceptable citations:
- Huang et al. (2024), "A foundation model for clinician-centered drug repurposing," *Nature Medicine*
- TxGNN GitHub repository (actual repo URL, not a summary page)
- PrimeKG paper and repository

News articles, blog posts, and press releases are NOT citations.

### 1.7 Write like a person
Short sentences. No "delve," "leverage," "robust framework," "seamless," "cutting-edge."
State what you did. State what happened. That is all.

---

## 2. The Six Research Questions

These map experiments to deliverables. Implement in this order.

---

### Q1 — Does KG augmentation improve over no-KG?

**Operationalization:**
- Same backbone (e.g., a 2-layer GNN or a small transformer encoder)
- Two conditions: (A) KG-augmented input, (B) text/ID-only input (no KG edges)
- Task: indication and contraindication prediction on PrimeKG
- Splits: standard split AND zero-shot split
- Metric: AUPRC (primary), AUROC (secondary)
- Seeds: [42, 0, 1] — report mean ± std

**Honest expected direction:** KG likely helps under zero-shot; gain under standard split may be small or absent. The experiment decides — not the expectation.

**Output files:**
- `results/metrics/gnn_no_kg_{split}_{seed}.json`
- `results/metrics/gnn_kg_{split}_{seed}.json`

---

### Q2 — Is there a better alternative to TxGNN's two-phase training?

**Operationalization:**
- Baseline: two-phase TxGNN (Phase 1 = self-supervised KG pretraining; Phase 2 = therapeutic task fine-tuning)
- Alternative A: single-stage multi-task (pretraining and task loss jointly from epoch 0)
- Alternative B: joint contrastive / metric-learning trained end to end
- Optional Alternative C: retrieval-augmented LLM + KG with no fine-tuning

**"Better" is defined upfront (before running):**
- Primary: higher AUPRC on zero-shot split at equal or lower total compute (wall-clock time on RTX 4060)
- Secondary: comparable AUPRC at materially lower compute (≥30% wall-clock reduction)
- Report the trade-off table. Do not declare a winner by feel.

**Output files:**
- `results/metrics/txgnn_two_phase_{split}_{seed}.json`
- `results/metrics/single_stage_{split}_{seed}.json`
- `results/metrics/joint_contrastive_{split}_{seed}.json`
- `results/metrics/compute_comparison.json` (wall-clock seconds per model)

---

### Q3 — Why is zero-shot prediction preferred for TxGNN?

**Two parts:**

**Conceptual (written once, not re-derived each run):**
Most diseases — especially rare ones — have no approved drugs in any training set.
A supervised per-disease classifier has no labels. Zero-shot generalization via
disease similarity is the only tractable path for rare disease coverage.

**Empirical:**
Plot a degradation curve: baseline AUPRC vs. number of training-time treatment edges per disease.
Show that performance collapses for diseases with zero edges (standard-split baseline)
while the TxGNN zero-shot model holds up.

**Output files:**
- `results/metrics/degradation_curve_data.json` (per-disease AUPRC vs. edge count)
- `results/figures/degradation_curve.png`

---

### Q4 — Two Case Studies

Pick before running. Do not change after seeing predictions.

**Disease 1 (zero-shot):** A rare disease with no approved therapy in PrimeKG training edges.
**Disease 2 (standard):** A well-studied disease with multiple approved therapies.

For each case study:
1. Run the model. Save raw predictions to `results/predictions/case_study_{disease}_{model}.csv`
2. Report the top-10 predicted drugs with their scores (from the file, not by hand)
3. Trace at least two predicted drugs through their KG paths (subgraph extraction)
4. Note any discrepancy between model prediction and known clinical evidence

The case studies are narrative but every predicted drug and every path comes from the model output files.

---

### Q5 — Comparative Table: GNN vs TxGNN

**A generated table, not assembled by hand.**

Requirements:
- Same data (PrimeKG)
- Same splits (standard and zero-shot)
- Same seeds ([42, 0, 1])
- Same metrics (AUPRC, AUROC for indication and contraindication)
- Columns: Model | Split | AUPRC_ind (mean±std) | AUROC_ind | AUPRC_contra | AUROC_contra | Wall-clock (s)

The table is produced by `scripts/generate_table.py` which reads from `results/metrics/`.
Do not type numbers into the report manually.

---

### Q6 — Is Attention Augmentation in TxGNN Optional?

**Hypothesis to test, not assert:**
TxGNN's zero-shot advantage comes from the disease-similarity / metric-learning module
operating over KG structure — not from an attention reweighting mechanism.

**Ablation design:**
- Condition A: TxGNN with attention (HAN/HGT-style neighbor reweighting) — ON
- Condition B: TxGNN backbone with attention — OFF (mean aggregation instead)
- Same training procedure, same splits, same seeds
- Measure delta in AUPRC on zero-shot split

**Decision rule (set before running):**
- Delta < 0.02 AUPRC → attention is optional (data supports the hypothesis)
- Delta ≥ 0.02 AUPRC → attention matters (report this instead; do not bury it)

**Output files:**
- `results/metrics/txgnn_attn_on_{split}_{seed}.json`
- `results/metrics/txgnn_attn_off_{split}_{seed}.json`

---

## 3. Folder Structure

```
BioInfo Finals/
├── CLAUDE.md                          ← This file
├── CONTEXT.md                         ← Changelog + hallucination log (update every session)
├── requirements.txt
├── data/
│   ├── raw/                           ← PrimeKG raw downloads (never modify)
│   ├── processed/                     ← Cleaned, featurized graph
│   └── splits/                        ← Train/val/test edge lists (standard + zero-shot)
├── src/
│   ├── data/
│   │   ├── primekg_loader.py
│   │   ├── splits.py
│   │   └── leakage_check.py
│   ├── models/
│   │   ├── gnn_baseline.py            ← Plain RGCN/HGT, no KG augmentation variant
│   │   ├── txgnn.py                   ← Scaled TxGNN reproduction
│   │   ├── txgnn_no_attn.py           ← Ablation: mean aggregation
│   │   ├── single_stage.py            ← Q2 alternative A
│   │   └── joint_contrastive.py       ← Q2 alternative B
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── finetune.py
│   │   └── single_stage_train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── leakage_check.py
│   │   └── zero_shot_eval.py
│   ├── case_studies/
│   │   └── case_study_runner.py
│   └── utils/
│       ├── gpu_config.py
│       └── logging_utils.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_gnn.ipynb
│   ├── 03_txgnn_reproduction.ipynb
│   ├── 04_ablations.ipynb
│   └── 05_case_studies.ipynb
├── configs/
│   ├── base.yaml
│   ├── gnn_baseline.yaml
│   ├── txgnn.yaml
│   ├── txgnn_no_attn.yaml
│   └── gpu.yaml
├── scripts/
│   ├── download_primekg.py
│   ├── run_baseline.py
│   ├── run_txgnn.py
│   ├── run_ablation.py
│   ├── run_leakage_check.py
│   └── generate_table.py
├── results/
│   ├── metrics/                       ← JSON per run: {model}_{split}_{seed}.json
│   ├── predictions/                   ← CSV per case study
│   └── figures/                       ← PNG plots generated by scripts
└── report/
    └── sections/
        ├── 01_introduction.md
        ├── 02_methods.md
        ├── 03_results.md
        ├── 04_case_studies.md
        └── 05_discussion.md
```

---

## 4. Results File Schema

Every `results/metrics/{model}_{split}_{seed}.json` must include:

```json
{
  "model": "txgnn_two_phase",
  "reproduction_type": "scaled_reproduction",
  "split": "zero_shot",
  "seed": 42,
  "gpu": "NVIDIA GeForce RTX 4060",
  "vram_gb": 8,
  "date": "YYYY-MM-DD",
  "wall_clock_seconds": 0,
  "indication": {
    "auprc": 0.0,
    "auroc": 0.0,
    "n_test_pairs": 0
  },
  "contraindication": {
    "auprc": 0.0,
    "auroc": 0.0,
    "n_test_pairs": 0
  },
  "notes": ""
}
```

`reproduction_type` must be one of: `"scaled_reproduction"`, `"paper_baseline"`, `"original_ablation"`.

---

## 5. Leakage Check (Phase 9 — Gates Everything)

Before reading any zero-shot result:

1. Load the zero-shot split: `data/splits/zero_shot_{train,val,test}.csv`
2. Extract the set of diseases in `test`
3. Assert that none of those diseases appear in any treatment edge in `train`
4. Write the assertion result to `results/metrics/leakage_check.json`:

```json
{
  "status": "PASS" | "FAIL",
  "leaking_diseases": [],
  "n_test_diseases": 0,
  "n_leaking": 0,
  "checked_at": "YYYY-MM-DDTHH:MM:SS"
}
```

5. If `status == "FAIL"`: stop. Fix the split. Rerun. Do not proceed with leaked splits.

---

## 6. What Claude Must Do Every Session

1. Read `CONTEXT.md` at the start of the session.
2. Before writing any new code or number, check if it conflicts with anything logged in `CONTEXT.md`.
3. At the end of the session (or after any significant code change), append to `CONTEXT.md`:
   - What was done (1–3 sentences, plain language)
   - What was assumed vs. what was verified
   - Any hallucination or deviation from the ground rules, labeled honestly
4. Never delete entries from `CONTEXT.md`. Append only.
5. If a result file does not exist yet, write `[NOT YET RUN]` in the report — not a placeholder number.
