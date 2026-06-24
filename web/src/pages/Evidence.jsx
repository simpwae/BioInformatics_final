import React from 'react'

const CITATIONS = [
  {
    id: 'huang2024',
    label: 'Huang et al. 2024 — TxGNN (Primary)',
    citation:
      'Huang K, Chandak P, Wang Q, Havaldar S, Vaid A, Leskovec J, Nadkarni GN, Glicksberg BS, Gehlenborg N, Zitnik M. "A foundation model for clinician-centered drug repurposing." Nature Medicine, 2024.',
    doi: 'https://doi.org/10.1038/s41591-024-03233-x',
    pmc: 'PMC11326339 (preprint), PMC11645266 (final)',
    usedFor: 'Q1–Q6 (primary paper being reproduced)',
    claimsUsed: [
      'Standard split TxGNN indication AUPRC: 0.91 ± 0.02 (Suppl. Table S1)',
      'Standard split HAN indication AUPRC: 0.87 ± 0.18 (Suppl. Table S1)',
      'Standard split TxGNN contraindication AUPRC: 0.82 ± 0.01 (Suppl. Table S2)',
      'Standard split HAN contraindication AUPRC: 0.84 ± 0.00 (Suppl. Table S2)',
      'Zero-shot TxGNN indication AUPRC: 0.90 ± 0.02 (Suppl. Table S1)',
      'Zero-shot TxGNN contraindication AUPRC: 0.80 ± 0.01 (Suppl. Table S2)',
      'Zero-shot relative gains: +19.0% ind / +23.9% contra vs next-best (Fig 2d, main text)',
      '92% of PrimeKG diseases have no indication edge (main text)',
      'Attention mechanism ineffective; replaced with fixed degree-based gating (lambda=0.7) (Methods)',
    ],
    notes: [
      'Zero-shot absolute values retrieved from Supplementary Tables S1/S2 (MOESM1 ESM)',
      '49.2% / 35.1% abstract headline: cannot be tied to a specific split in main text; not used in tables',
      'Standard contraindication: HAN (0.84) beats TxGNN (0.82) — TxGNN gain claim refers to indications only',
    ],
  },
  {
    id: 'chandak2023',
    label: 'Chandak et al. 2023 — PrimeKG',
    citation:
      'Chandak P, Huang K, Zitnik M. "Building a knowledge graph to enable precision medicine." Scientific Data, 10, 67 (2023).',
    doi: 'https://doi.org/10.1038/s41597-023-01960-3',
    usedFor: 'Data description (PrimeKG)',
    claimsUsed: [
      '129,375 nodes, 8,100,498 edges (our downloaded data)',
      'Harvard Dataverse download: doi:10.7910/DVN/IXA7BM',
    ],
  },
  {
    id: 'kgbert2019',
    label: 'Yao et al. 2019 — KG-BERT',
    citation:
      'Yao L, Mao C, Luo Y. "KG-BERT: BERT for Knowledge Graph Completion." arXiv:1909.03193, 2019.',
    doi: 'https://arxiv.org/abs/1909.03193',
    usedFor: 'Background — LM+KG lineage (Q1, Q2)',
    claimsUsed: [
      'Treats KG triples as text sequences; classifies valid/invalid',
      'Competitive on WN18RR, FB15k-237 triple classification',
    ],
    notes: ['Specific accuracy numbers not confirmed from PDF — background context only'],
  },
  {
    id: 'dragon2022',
    label: 'Yasunaga et al. 2022 — DRAGON',
    citation:
      'Yasunaga M, Bosselut A, Ren H, et al. "Deep Bidirectional Language-Knowledge Graph Pretraining." NeurIPS 2022.',
    doi: 'https://arxiv.org/abs/2210.09338',
    usedFor: 'Background — joint LM+KG pretraining (Q1, Q2)',
    claimsUsed: ['+5% average gain on downstream QA tasks (abstract)'],
    notes: [
      'KG is ConceptNet/UMLS, not PrimeKG. Numbers do NOT transfer to drug repurposing setting.',
    ],
  },
]

export default function Evidence() {
  return (
    <div className="page">
      <h1>Evidence and Sources</h1>
      <p>
        Every number that appears in this project links to either a result file in{' '}
        <code>results/</code> or a primary paper citation here. This page documents all external
        sources and which claims are taken from them.
      </p>

      <div className="warn-box">
        <p>
          <strong>Provenance rule:</strong> Paper numbers go in the{' '}
          <code>paper_reported</code> column, labeled with the source. Scaled reproduction numbers
          go in the <code>scaled_reproduction</code> column. They are never merged or compared
          without explicit labeling.
        </p>
      </div>

      {CITATIONS.map((c) => (
        <section
          key={c.id}
          className="section"
          aria-labelledby={`cite-${c.id}`}
          style={{ borderTop: '1px solid #d0d0d0', paddingTop: '1.5rem' }}
        >
          <h2 id={`cite-${c.id}`}>{c.label}</h2>
          <p>
            <strong>Citation:</strong> {c.citation}
          </p>
          {c.doi && (
            <p>
              <strong>DOI:</strong>{' '}
              <a href={c.doi} target="_blank" rel="noopener noreferrer">
                {c.doi}
              </a>
            </p>
          )}
          {c.pmc && (
            <p>
              <strong>PMC:</strong> {c.pmc}
            </p>
          )}
          <p>
            <strong>Used for:</strong> {c.usedFor}
          </p>
          <h3>Claims used</h3>
          <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            {c.claimsUsed.map((claim, i) => (
              <li key={i}>{claim}</li>
            ))}
          </ul>
          {c.notes && c.notes.length > 0 && (
            <>
              <h3>Notes and caveats</h3>
              <ul style={{ paddingLeft: '1.5rem' }}>
                {c.notes.map((note, i) => (
                  <li key={i} style={{ color: '#555' }}>
                    {note}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      ))}

      <section
        className="section"
        aria-labelledby="result-files"
        style={{ borderTop: '1px solid #d0d0d0', paddingTop: '1.5rem' }}
      >
        <h2 id="result-files">Result Files (scaled reproduction)</h2>
        <p>
          All metric files follow the schema:{' '}
          <code>results/metrics/{'{model}_{split}_{seed}.json'}</code>. Key files:
        </p>
        <table>
          <thead>
            <tr>
              <th>File</th>
              <th>Content</th>
              <th>Serves</th>
            </tr>
          </thead>
          <tbody>
            {[
              ['results/metrics/comparison_table.csv', 'Full model comparison', 'Q1, Q2, Q5'],
              ['results/metrics/q6_ablation_table.csv', 'Attention ablation AUPRC', 'Q6'],
              ['results/ablations/matrix.json', 'Full ablation matrix + Q6 decision', 'Q6'],
              ['results/metrics/degradation_curve_data.json', 'Per-disease AUPRC vs edge count', 'Q3'],
              [
                'results/predictions/case_study_caseA_txgnn.csv',
                'FHC top-20 drug predictions',
                'Q4',
              ],
              [
                'results/predictions/case_study_caseB_txgnn.csv',
                'S. aureus top-20 drug predictions',
                'Q4',
              ],
              ['results/predictions/case_study_caseA_paths_txgnn.csv', 'FHC KG paths', 'Q4'],
              [
                'results/predictions/case_study_caseB_paths_txgnn.csv',
                'S. aureus KG paths',
                'Q4',
              ],
              [
                'results/metrics/leakage_check_seed{n}.json',
                'Leakage check (PASS for seeds 42, 0, 1)',
                'All zero-shot results',
              ],
            ].map(([file, content, serves], i) => (
              <tr key={i}>
                <td>
                  <code>{file}</code>
                </td>
                <td>{content}</td>
                <td>{serves}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section
        className="section"
        aria-labelledby="not-used"
        style={{ borderTop: '1px solid #d0d0d0', paddingTop: '1.5rem' }}
      >
        <h2 id="not-used">Claims NOT Used (and Why)</h2>
        <table>
          <thead>
            <tr>
              <th>Claim</th>
              <th>Reason not used</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>+49.2% / +35.1% headline (Huang et al. 2024 abstract)</td>
              <td>
                Cannot be tied to any specific split in the main text of either version.
                Appears in abstract only; unresolvable from article body.
              </td>
            </tr>
            <tr>
              <td>DRAGON +5% / +10% (QA tasks)</td>
              <td>
                Different domain (QA, ConceptNet) — does not transfer to PrimeKG drug repurposing.
              </td>
            </tr>
            <tr>
              <td>QA-GNN accuracy numbers</td>
              <td>Not confirmed from PDF; background context only, different domain.</td>
            </tr>
            <tr>
              <td>transformer_kg / transformer_nokg results</td>
              <td>[NOT YET RUN] — filtered out from comparison table.</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  )
}
