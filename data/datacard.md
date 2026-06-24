# PrimeKG Data Card

## Identity

| Field | Value |
|---|---|
| Name | PrimeKG (Precision Medicine Knowledge Graph) |
| Version | Original release (kg.csv from Harvard Dataverse) |
| Paper | Chandak P, Huang K, Zitnik M. "Building a knowledge graph to enable precision medicine." *Scientific Data* 10, 67 (2023). DOI: 10.1038/s41597-023-01960-3 |
| Source | Harvard Dataverse: https://dataverse.harvard.edu/api/access/datafile/6180620 |
| Downloaded | [DATE TO BE FILLED BY DOWNLOAD SCRIPT] |
| License | MIT (codebase). Individual constituent datasets retain their own licenses (see below). |

## Provenance

PrimeKG integrates 20 biomedical databases (14 confirmed from GitHub README; full list in paper):
Bgee, Comparative Toxicogenomics Database, DisGeNET, DrugBank, DrugCentral, Entrez Gene,
Gene Ontology, Human Phenotype Ontology, MONDO, OMIM, Reactome, SIDER, UBERON, UMLS.
Remaining 6 sources unconfirmed (paper PDF inaccessible at time of writing — see lit/notes.md).

## Statistics (computed from downloaded data — filled by scripts/download_primekg.py)

These values are placeholders until the download completes and compute_stats() runs.

| Statistic | Value |
|---|---|
| Total edges (kg.csv rows) | 8,100,498 |
| Total nodes (nodes.tab) | 129,375 |
| Number of relation types | 30 |
| Number of diseases | 17,080 |
| Number of drugs / therapeutic candidates | 7,957 |
| Indication edges | 18,776 |
| Contraindication edges | 61,350 |
| Diseases with any treatment edge | 4,005 |
| Diseases with NO treatment edges (zero-shot pool) | 13,075 |

**Edge count discrepancy vs. paper:** Chandak et al. (2023) report 4,050,249 relationships.
The downloaded kg.csv contains 8,100,498 rows. Spot-checking shows drug_drug edges are NOT
simply stored twice (only ~1.2% of a sample had reversed pairs), so this is likely a dataset
update or a different counting convention (directed vs. undirected). Do NOT use 4,050,249 in
any result; use 8,100,498 from the actual file.

Full breakdown by node type and relation type: `data/primekg_stats.json`

## Files

| File | Description |
|---|---|
| `data/raw/kg.csv` | Full edge list. Columns: x_id, x_type, y_id, y_type, relation, display_relation |
| `data/raw/nodes.tab` | Node metadata |
| `data/raw/disease_features.tab` | Disease-level features |
| `data/raw/drug_features.tab` | Drug-level features |
| `data/splits/standard/seed_{n}/` | Standard split (random edge holdout) |
| `data/splits/zeroshot/seed_{n}/` | Zero-shot split (disease holdout) |

## Preprocessing

1. No filtering applied to non-therapeutic edges.
2. For splits, only `indication` and `contraindication` edges are partitioned.
3. All other edges (protein–protein, gene–disease, etc.) remain in the training graph for all splits.
4. Encoding: original node IDs are mapped to integer indices per node type (see `src/data/primekg_loader.py`).

## Split Design

### Standard Split (seed-dependent; canonical seed=42)
- 10% of therapeutic edges randomly held out as test.
- 10% of remaining as validation.
- Test diseases MAY have other treatment edges in training.
- Not zero-shot.

### Zero-Shot Split (seed-dependent; canonical seed=42)
- 20% of all diseases (by disease ID) are held out entirely.
- All therapeutic edges from held-out diseases go to val (20% of held-out) or test (80% of held-out).
- Zero-shot guarantee: no held-out disease has ANY treatment edge in training.
- Verified by `src/data/leakage_check.py` (must PASS before results are accepted).

## Known Limitations

1. PrimeKG reflects the state of biomedical databases at its time of construction (~2022).
   It does not include drugs approved or indications granted after that date.
2. DrugBank and SIDER have restrictive licenses for commercial use; academic use is permitted.
3. Some constituent databases have known incompleteness (negative results underreported in SIDER;
   rare disease coverage in DisGeNET is better than common disease coverage in some areas).
4. The zero-shot split separates diseases but not drugs. A drug appearing in the test set
   may have been seen in training in a different disease context.
