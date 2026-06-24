# Literature Notes

> Sourcing discipline: numbers come from the source labeled. Where the primary paper PDF was
> inaccessible (compressed binary / paywall redirect), this is flagged explicitly.
> Do not treat flagged numbers as verbatim paper quotes — verify against the actual paper.

---

## Paper 1 — TxGNN (Primary: the paper this project reproduces)

**Citation:** Huang K, Chandak P, Wang Q, Havaldar S, Vaid A, Leskovec J, Nadkarni GN,
Glicksberg BS, Gehlenborg N, Zitnik M. "A foundation model for clinician-centered drug
repurposing." *Nature Medicine*, 2024. DOI: 10.1038/s41591-024-03233-x

**Primary sources:**
- Preprint: PMC11326339 (medRxiv v4, open-access via NIH preprint pilot) — RETRIEVED AND READ
- Peer-reviewed: PMC11645266 (*Nature Medicine* final version) — MAIN TEXT READ; supplementary file (MOESM1 ESM, "41591_2024_3233_MOESM1_ESM.pdf") not retrieved (CAPTCHA-blocked; requires direct browser download)

**One-line takeaway:** A GNN pretrained on a biomedical KG with a disease-similarity metric-learning
module predicts drug indications and contraindications for diseases with no approved therapy at train
time (zero-shot), and outperforms prior GNN baselines substantially on that setting.

---

### What the paper actually claims (from PMC11326339 — VERIFIED primary source)

#### Architecture
- Two-phase training:
  - Phase 1 (`pretrain()`): self-supervised link prediction over all edge types in the KG.
  - Phase 2 (`finetune()`): drug-disease relation prediction with a disease-similarity metric-learning module.
- Gating mechanism for combining similarity-propagated embeddings: **fixed degree-based exponential gating, NOT learnable attention.** Exact quote from main paper (s41591-024-03233-x.pdf, p.15, Methods): *"Use of a learnable attention mechanism is ineffective, because it overvalues the original embeddings for well-represented diseases, neglecting the auxiliary embedding. Alternatively, we introduced an approach that determines weighting based on the degree of node connectivity |𝒩ᵣᵢ| of the queried drug–disease pair."* The scalar uses an inflated exponential distribution density function (λ parameter specified in the equation; lambda=0.7 cited from PMC11326339 preprint where formula text was extractable). Source: Methods, both versions.
- Pre-trained on PrimeKG: 17,080 clinically-recognized diseases, 7,957 therapeutic candidates.

#### KG statistics — TWO SOURCES, ONE DISCREPANCY (document honestly)

**Main text (s41591-024-03233-x.pdf, p.14):** 123,527 nodes, 8,063,026 edges, 10 node types, 29 edge types.

**Supplementary Table S8 (MOESM1 ESM, p.35):** 129,375 total nodes. Breakdown:
biological_process 28,642 | protein 27,671 | disease 17,080 | phenotype 15,311 | anatomy 14,035 | molecular_function 11,169 | drug 7,957 | cellular_component 4,176 | pathway 2,516 | exposure 818.

Our downloaded kg.csv: **129,375 nodes** (matches Supplementary Table S8 exactly).

Reconciliation: The main text figure (123,527) is smaller than the supplementary table (129,375). Likely the main text counts only nodes that appear in at least one edge in the processed/filtered graph, while the supplementary table counts all nodes including isolated ones. The supplementary Table S8 is the authoritative breakdown; the main text is the filtered-graph count. Both are from the same paper. Our data matches the supplementary figure.

- Indications: 9,388 unique pairs; contraindications: 30,675 unique pairs (paper, undirected). Our kg.csv: 18,776 indication rows, 61,350 contraindication rows (directed, each pair twice) — reconciled: 18,776/2 = 9,388 ✓
- 92% of diseases have no indication edge (confirmed: main paper text)

#### Standard split (test diseases *can* have training treatment edges) — VERIFIED from PMC11326339
- TxGNN AUPRC (indications): **0.913**
- HAN (best GNN-family baseline): **0.873**
- Relative gain: +4.3% over HAN
- Note: HAN, HGT, and RGCN are all in the GNN family. HAN was the best GNN baseline for standard-split indications. A plain RGCN or no-message-passing model would score lower; our `gnn_no_kg` is not a direct comparator to HAN.

#### Zero-shot split — VERIFIED: absolute values NOW RETRIEVED from Suppl. Table S1/S2 (MOESM1 ESM)

**Supplementary Table S1 — Indication AUPRC (mean ± std across 5 seeds unless noted):**

| Split | RGCN | HAN | HGT | BioBERT | TxGNN |
|-------|------|-----|-----|---------|-------|
| Random (standard) | 0.84±0.02 | 0.87±0.18 | 0.72±0.20 | 0.81±0.02 | **0.91±0.02** |
| Zero-shot (random) | 0.50±0.04 | 0.62±0.03 | 0.56±0.08 | 0.76±0.03 | **0.90±0.02** |
| Cell Proliferation | 0.47±0.00 | 0.65±0.04 | 0.61±0.10 | 0.83±0.0 | **0.92±0.00** |
| Mental Health | 0.55±0.01 | 0.61±0.03 | 0.56±0.04 | 0.64±0.0 | **0.89±0.00** |
| Cardiovascular | 0.41±0.00 | 0.63±0.05 | 0.60±0.05 | **0.66±0.0** | 0.64±0.00 |
| Anemia | 0.42±0.00 | 0.67±0.01 | 0.53±0.05 | 0.64±0.0 | **0.96±0.00** |
| Adrenal Gland | 0.38±0.00 | 0.59±0.04 | 0.54±0.10 | 0.64±0.0 | **0.99±0.00** |
| Autoimmune | 0.51±0.00 | 0.53±0.03 | 0.49±0.05 | 0.66±0.0 | **0.87±0.00** |
| Metabolic Disorder | 0.67±0.01 | 0.56±0.08 | 0.54±0.05 | 0.68±0.0 | **0.76±0.00** |
| Diabetes | 0.40±0.00 | 0.67±0.11 | 0.62±0.17 | 0.69±0.0 | **0.87±0.00** |
| Neurodegenerative | **0.84±0.00** | 0.71±0.10 | 0.60±0.06 | 0.57±0.0 | 0.95±0.00 |

**Supplementary Table S2 — Contraindication AUPRC:**

| Split | RGCN | HAN | HGT | BioBERT | TxGNN |
|-------|------|-----|-----|---------|-------|
| Random (standard) | 0.77±0.02 | **0.84±0.00** | 0.63±0.19 | 0.58±0.02 | 0.82±0.01 |
| Zero-shot (random) | 0.64±0.03 | 0.50±0.03 | 0.54±0.06 | 0.57±0.03 | **0.80±0.01** |
| Cell Proliferation | 0.54±0.00 | 0.51±0.04 | 0.47±0.03 | **0.78±0.0** | 0.87±0.00 |
| Mental Health | 0.67±0.00 | 0.44±0.02 | 0.56±0.04 | 0.54±0.0 | **0.79±0.00** |
| Cardiovascular | 0.63±0.00 | 0.47±0.02 | 0.51±0.02 | 0.56±0.0 | **0.72±0.00** |
| Anemia | 0.65±0.00 | 0.54±0.04 | 0.51±0.04 | 0.53±0.0 | **0.74±0.00** |
| Adrenal Gland | 0.75±0.00 | 0.46±0.02 | 0.55±0.08 | 0.47±0.0 | **0.88±0.00** |
| Autoimmune | 0.68±0.00 | 0.40±0.01 | 0.61±0.08 | 0.40±0.0 | **0.86±0.00** |
| Metabolic Disorder | 0.58±0.00 | 0.56±0.00 | 0.51±0.03 | 0.53±0.0 | **0.78±0.00** |
| Diabetes | 0.57±0.00 | 0.56±0.03 | 0.52±0.05 | 0.51±0.0 | **0.69±0.00** |
| Neurodegenerative | 0.71±0.00 | 0.42±0.01 | 0.55±0.04 | 0.60±0.0 | **0.78±0.00** |

**Key observations from the tables:**
- Zero-shot indication: TxGNN **0.90±0.02** vs BioBERT 0.76±0.03 → (0.90-0.76)/0.76 = 18.4% ≈ **+19.0%** ✓ (Fig 2d gain reconciled)
- Zero-shot contraindication: TxGNN **0.80±0.01** vs RGCN 0.64±0.03 → (0.80-0.64)/0.64 = 25.0% (paper says 23.9%; consistent with rounding at 2 decimal places)
- Standard contraindication: **HAN (0.84) beats TxGNN (0.82)**. TxGNN's gain claim (main text) refers to indications only. Standard contraindication is not TxGNN's advantage.
- Cardiovascular indication: TxGNN 0.64 < BioBERT 0.66 — TxGNN is not best on every disease area.
- The 49.2% / 35.1% headline cannot be derived from any row or combination in these tables. Confirmed unreconcilable.

Nine disease-area splits: indication gains 0.5%–59.3% (mean 25.72%); contraindication gains 11.8%–35.6% (mean 18.67%). Confirmed in both versions.

#### 49.2% / 35.1% discrepancy — CONFIRMED SETTLED (both versions read)
- The peer-reviewed abstract (PMC11645266) states +49.2% (indications), +35.1% (contraindications) for "stringent zero-shot evaluation."
- No figure or paragraph in the main text of either version maps a number to exactly 49.2 / 35.1. The in-text zero-shot gains are 19.0/23.9 (Fig 2d, random zero-shot) and the disease-area ranges above.
- **49.2 / 35.1 is the abstract's headline aggregate — it cannot be tied to a specific split from the article body. Do not use as a per-split figure in the paper_reported table.**
- If cited at all, label it: "Abstract headline, Huang et al. 2024 Nat Med; split not specified in main text."

#### Explainer faithfulness (from PMC11326339)
- Top ~14.9% of edges by importance: AUPRC = 0.886 (vs 0.890 full graph)
- Remaining ~85.1% of edges alone: AUPRC = 0.628

---

### Caveats the authors give
- Model cannot be verified against real-world clinical trial outcomes.
- Zero-shot performance is measured on held-out diseases from PrimeKG, not prospective clinical data.
- "Up to 49.2% higher accuracy" does not reconcile to a named split in the preprint text — origin unclear.

### Relevance to Q1–Q6

| Q | Relevance |
|---|-----------|
| Q1 | Primary source. Standard split: TxGNN 0.913 vs HAN 0.873. Zero-shot: +19.0% / +23.9% vs next-best (relative). |
| Q2 | Two-phase training (pretrain + finetune) is the baseline procedure to compare against. |
| Q3 | Paper's framing: zero-shot preferred because most diseases, especially rare ones, have no labeled drugs. 92% of PrimeKG diseases have no indication edge. |
| Q4 | This paper's output predictions are what the case studies should use. |
| Q5 | Standard: TxGNN 0.913, HAN 0.873 (main text, both versions). Zero-shot absolutes: only in Suppl. S1/S2 (SI file, not readable online); relative gains (+19.0%/+23.9%) are the best citable main-text figures. |
| Q6 | The paper's gating step uses FIXED degree-based exponential gating (λ=0.7), not learnable attention — because learnable attention was found ineffective. This independently supports our ablation finding. |

---

## Paper 2 — PrimeKG

**Citation:** Chandak P, Huang K, Zitnik M. "Building a knowledge graph to enable precision
medicine." *Scientific Data*, 10, 67 (2023). DOI: 10.1038/s41597-023-01960-3

**One-line takeaway:** PrimeKG integrates 20 biomedical databases into a heterogeneous graph with
~129K nodes and ~4M edges across 10 biological scales; released under MIT license with Harvard
Dataverse hosting.

---

### What the paper actually claims

#### Graph statistics (from GitHub README and web sources; paper PDF paywalled)
- **Total nodes:** ~129,375 (figure from web search; not verified against paper table)
  - Diseases: 17,080 (confirmed multiple sources)
  - Drugs / therapeutic candidates: 7,957 (confirmed from TxGNN GitHub)
  - Other node types: genes/proteins, biological processes, pathways, anatomical terms, phenotypes (exact counts NOT confirmed from primary paper)
- **Total edges:** 4,050,249 relationships (confirmed multiple sources)
- **Relation types:** 29 relation types listed in GitHub (not all with counts confirmed)
- **Biological scales covered:** disease-protein perturbations, biological processes, pathways, anatomical, phenotypic, drug therapeutic action (10 scales stated in paper abstract)

#### Data sources (14 confirmed from README; paper states 20 resources)
Bgee, Comparative Toxicogenomics Database, DisGeNET, DrugBank, DrugCentral, Entrez Gene,
Gene Ontology, Human Phenotype Ontology, MONDO, OMIM, Reactome, SIDER, UBERON, UMLS.
⚠ Full list of 20 not confirmed. These 14 are from the GitHub README.

#### Download
- Main file: `kg.csv` from Harvard Dataverse
- Command: `wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620`
- Additional files: `nodes.tab`, `edges.csv`, `disease_features.tab`, `drug_features.tab`
- License: MIT (codebase). Individual source datasets retain their own licenses.

---

## Paper 3 — KG-BERT

**Citation:** Yao L, Mao C, Luo Y. "KG-BERT: BERT for Knowledge Graph Completion."
arXiv:1909.03193, 2019.

**One-line takeaway:** Fine-tuning BERT on (head, relation, tail) triples as text sequences achieves
competitive link prediction but is too slow for practical ranking (must score all entity pairs).

---

### What the paper actually claims
- Treats each triple as a sentence: "[CLS] head_text [SEP] relation_text [SEP] tail_text [SEP]"
- BERT classifies the triple as valid/invalid; score is used to rank candidates.
- Evaluated on WN18RR, FB15k-237, UMLS for triple classification and link prediction.
- Claims "state-of-the-art performance" on triple classification and link prediction on tested benchmarks.
- Specific numbers: Hits@10 and MRR values on WN18RR and FB15k-237 (not confirmed verbatim; PDF accessible but not fetched).
- ⚠ Inference is O(N_entities) per query — impractical for large KGs without candidate pruning.

### Relevance to Q1–Q6
| Q | Relevance |
|---|-----------|
| Q1 | Establishes the LM+KG lineage: text-informed scoring of KG triples is one approach to augmenting a model with KG structure. |
| Q2 | Fine-tuning a pre-trained LM on KG triples (single-stage, no graph pretraining) is a relevant baseline strategy for Q2 alternatives. |

---

## Paper 4 — DRAGON

**Citation:** Yasunaga M, Bosselut A, Ren H, Zhang X, Manning CD, Liang PS, Leskovec J.
"Deep Bidirectional Language-Knowledge Graph Pretraining." *NeurIPS 2022*.
arXiv:2210.09338

**One-line takeaway:** Jointly pretraining an LM with a GNN over a KG (bidirectional cross-modal
fusion) outperforms LM-only pretraining by +5% average and +10% on complex reasoning tasks —
but the KG is ConceptNet/UMLS for QA, not a drug-disease graph.

---

### What the paper actually claims (from abstract page and web summaries)
- Prior LM+KG works "learn only a shallow combination of text and KG" — DRAGON proposes deep bidirectional fusion at pretraining time.
- Results (from arXiv abstract):
  - "+5% absolute gain on average" across downstream QA tasks vs baselines.
  - "+10% on questions involving long contexts or multi-step reasoning."
  - "+8% on OBQA and RiddleSense."
  - State-of-the-art on "various BioNLP tasks."
  - ⚠ Specific accuracy numbers per task not confirmed (PDF binary-encoded).
- Task setting: question answering (CommonsenseQA, OBQA, RiddleSense, BioNLP); not drug repurposing.

### Relevance to Q1–Q6
| Q | Relevance |
|---|-----------|
| Q1 | Strongest evidence that deep LM+KG pretraining outperforms LM-only — but on QA tasks. Whether this transfers to drug repurposing link prediction is NOT established by this paper. |
| Q2 | DRAGON's "single pretraining + downstream fine-tune" is structurally different from TxGNN's two-phase. It's a relevant alternative architecture but uses different modalities. |

### Critical caveat
DRAGON's KG is ConceptNet (commonsense) + UMLS; the text is Wikipedia. It is not a biomedical drug-disease KG. The +5% / +10% numbers do not apply to the PrimeKG + therapeutic task setting. Do not transfer these numbers to Q1 claims.

---

## Paper 5 — QA-GNN

**Citation:** Yasunaga M, Ren H, Bosselut A, Liang P, Leskovec J. "QA-GNN: Reasoning with
Language Models and Knowledge Graphs for Question Answering." *NAACL 2021*.
arXiv:2104.06378

**One-line takeaway:** Connecting an LM's QA context to a KG subgraph and jointly updating both
with a GNN consistently beats LM-only on commonsense and biomedical QA benchmarks — but the
margin depends on task and the KG is ConceptNet/UMLS, not a drug-disease graph.

---

### What the paper actually claims (from abstract; numbers not confirmed from tables)
- Two innovations: (i) relevance scoring of KG nodes by LM, (ii) joint GNN reasoning over QA + KG graph.
- "Outperforms existing LM and LM+KG models" on CommonsenseQA, OpenBookQA, MedQA-USMLE.
- Specific Accuracy numbers: NOT confirmed (PDF binary, table numbers not retrieved).

### Relevance to Q1–Q6
| Q | Relevance |
|---|-----------|
| Q1 | Supports the general principle that KG augmentation helps LMs, but in QA not drug repurposing. |
| Q2 | QA-GNN's joint (single-stage) LM+GNN reasoning is architecturally related to Q2 Alternative A (single-stage multi-task). |

### Critical caveat
Same domain mismatch as DRAGON: CommonsenseQA / UMLS ≠ PrimeKG drug repurposing. Cannot claim Q1 is answered by QA-GNN results.

---

## Paper 6 — StAR

**Citation:** Wang B, Shen T, Long G, Zhou T, Wang Y, Chang Y. "Structure-Augmented Text
Representation Learning for Efficient Knowledge Graph Completion." *WWW 2021*.
arXiv:2004.14781

**One-line takeaway:** A Siamese text encoder combining deterministic classification with structural
(spatial) measurement reaches Hits@10 ≈ 62.4 on WN18RR and ≈ 46.4 on FB15k-237 — competitive
with pure structure methods at much lower cost; but again on general KGs, not biomedical.

---

### What the paper actually claims
- Hits@10 on WN18RR: 62.3–62.4 (confirmed from web search result citing paper table)
- Hits@10 on FB15k-237: 45.8–46.4 (confirmed from web search result)
- Claim: text-based methods can match structure-only methods without full entity embedding tables.
- ⚠ ATOMIC results: 12.5–12.6 Hits@10 (from same web search; caveat applies).

### Relevance to Q1–Q6
| Q | Relevance |
|---|-----------|
| Q1 | Establishes the KGC (structure + text) lineage. Shows structure augmentation helps text representations on benchmark KGs. |

---

## Claims Register — Sourced Evidence per Research Question

| Q | Claim | Source | Source Type | Caveat |
|---|-------|--------|-------------|--------|
| Q1 | TxGNN standard split indication AUPRC = 0.913 | PMC11326339 main text | Primary (preprint) | Confirmed from actual paper |
| Q1 | HAN (best GNN baseline) standard split indication AUPRC = 0.873 | PMC11326339 main text | Primary (preprint) | Confirmed |
| Q1 | TxGNN zero-shot random split: +19.0% ind, +23.9% contra over next-best | PMC11326339 Fig 2d | Primary (preprint) | Relative gains only; absolute values in Suppl. S1/S2 |
| Q1 | +49.2% / +35.1% headline figures: split not named in preprint text | PMC11326339 abstract | Primary (preprint) | Do NOT use in table — cannot assign to a specific split |
| Q1 | DRAGON +5% avg over LM-only (QA tasks) | arXiv abstract | Abstract-level | Different domain (QA, not drug repurposing) |
| Q2 | Two-phase training: Phase 1 = pretrain(); Phase 2 = finetune() metric learning | PMC11326339 Methods + GitHub | Primary | Confirmed |
| Q3 | Zero-shot needed because 92% of PrimeKG diseases have no indication edge | PMC11326339 | Primary (preprint) | Confirmed; exact figure "92%" from paper |
| Q4 | 17,080 diseases, 7,957 drugs in PrimeKG | PMC11326339 Methods | Primary (preprint) | Confirmed |
| Q5 | Paper absolute zero-shot AUPRC values | PMC11326339 Suppl. S1/S2 | Primary (preprint) | NOT YET RETRIEVED |
| Q6 | Paper gating uses fixed degree-based exponential function (λ=0.7), not learnable attention; learnable attention found ineffective | PMC11326339 Methods | Primary (preprint) | Confirmed — independently supports our ablation finding |

---

## What is NOT Known (honest gaps, updated 2026-06-24)

1. **Absolute AUPRC for TxGNN and baselines on zero-shot split. — RESOLVED.**
   Supplementary Tables S1 and S2 (MOESM1 ESM) retrieved and read directly. Full tables are now in the "Zero-shot split" section above. The paper_reported column in 03_results.md is filled: TxGNN zero-shot indication = 0.90±0.02, contraindication = 0.80±0.01. No remaining gap.

2. **Node count discrepancy: paper says 123,527 nodes; our kg.csv has 129,375.**
   Likely version difference or isolated nodes excluded from paper's filtered graph. No impact on results. Note in data card only.

3. **QA-GNN and DRAGON accuracy numbers per task.**
   Background context only, not primary benchmarks. No action needed.

4. **49.2% / 35.1% abstract headline — settled as unresolvable from article body.**
   Both versions confirmed: these figures appear in the abstract but no main-text figure or paragraph assigns them to a specific split. Do not use as a per-split number. If cited: "Abstract headline, Huang et al. 2024 Nat Med; split not specified in main text."
