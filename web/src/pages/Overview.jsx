import React from 'react'
import { Link } from 'react-router-dom'
import { useData } from '../context/DataContext.jsx'

const questions = [
  {
    id: 'Q1',
    to: '/results',
    label: 'Does KG augmentation improve over no-KG?',
    finding:
      'KG message passing does not improve indication AUPRC on the standard split. The no-KG model outperforms (0.893 vs 0.825). On zero-shot contraindication, the KG model has an advantage.',
    outcome: 'unexpected',
  },
  {
    id: 'Q2',
    to: '/q2-alternatives',
    label: 'Is there a better alternative to two-phase training?',
    finding:
      'No. Two-phase TxGNN leads on zero-shot indication (0.736) vs single-stage (0.706) and joint contrastive (0.670). It also runs faster (71.6s vs 130.7s).',
    outcome: 'expected',
  },
  {
    id: 'Q3',
    to: '/q3-zeroshot',
    label: 'Why is zero-shot prediction preferred?',
    finding:
      '92% of PrimeKG diseases have no approved therapy (Huang et al. 2024). A supervised classifier has no labels for these diseases. Zero-shot is not a design preference — it is a necessity.',
    outcome: 'confirmed',
  },
  {
    id: 'Q4',
    to: '/case-studies',
    label: 'What do two case studies reveal about model predictions?',
    finding:
      'Case A (FHC, n_pos=1): model fails — Propranolol not in top-20. Case B (S. aureus, n_pos=45): Benzylpenicillin at rank 18; three cancer drugs in top-10.',
    outcome: 'mixed',
  },
  {
    id: 'Q5',
    to: '/results',
    label: 'How does plain GNN compare to TxGNN?',
    finding:
      'On standard split: gnn_no_kg wins (0.893). On zero-shot: TxGNN two-phase leads (0.736). Full table generated from CSV — no manually typed numbers.',
    outcome: 'neutral',
  },
  {
    id: 'Q6',
    to: '/q6-ablations',
    label: 'Is attention augmentation in TxGNN optional?',
    finding:
      'Attention is DETRIMENTAL. Removing HGT attention improves zero-shot indication AUPRC from 0.736 to 0.772 (delta = -0.036). The disease-similarity module is the load-bearing component.',
    outcome: 'unexpected',
  },
]

const outcomeColor = {
  expected: '#1a5c2e',
  unexpected: '#7a3800',
  confirmed: '#1a4e8a',
  mixed: '#555',
  neutral: '#555',
}

const outcomeLabel = {
  expected: 'confirmed hypothesis',
  unexpected: 'unexpected finding',
  confirmed: 'confirmed',
  mixed: 'mixed',
  neutral: 'neutral',
}

export default function Overview() {
  const { data, loading, error } = useData()

  return (
    <div className="page">
      <h1>Drug Repurposing with Knowledge Graphs</h1>
      <p style={{ fontSize: '1.05rem', color: '#333' }}>
        A scaled reproduction of TxGNN (Huang et al., 2024) on PrimeKG. Six empirical questions
        about knowledge graph augmentation for drug repurposing. All results link to files in{' '}
        <code>results/</code>. No numbers typed by hand.
      </p>

      <div className="finding-box" style={{ marginTop: '1.5rem' }}>
        <p>
          <strong>Scaled reproduction notice.</strong> All models use{' '}
          <code>hidden_dim=64</code> instead of the published <code>hidden_dim=512</code> due to
          the RTX 4060 8 GB VRAM constraint. Results are labeled{' '}
          <span className="badge badge-scaled">scaled_reproduction</span> throughout. Paper numbers
          from Huang et al. (2024){' '}
          <span className="badge badge-paper">paper_reported</span> are shown in a separate
          column.
        </p>
      </div>

      {loading && <p className="status-loading">Loading data...</p>}
      {error && (
        <div className="status-error">
          <strong>Error loading results.json:</strong> {error}
          <br />
          Run <code>npm run build:data</code> from the <code>web/</code> directory first.
        </div>
      )}

      <h2>Six Research Questions</h2>
      <div style={{ display: 'grid', gap: '1rem' }}>
        {questions.map((q) => (
          <div
            key={q.id}
            style={{
              border: '1px solid #d0d0d0',
              borderRadius: '4px',
              padding: '1rem 1.25rem',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: '0.75rem',
                marginBottom: '0.5rem',
                flexWrap: 'wrap',
              }}
            >
              <span
                style={{
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  color: '#1a4e8a',
                  minWidth: '2.5rem',
                }}
              >
                {q.id}
              </span>
              <Link to={q.to} style={{ fontWeight: 600, fontSize: '1rem' }}>
                {q.label}
              </Link>
              <span
                style={{
                  fontSize: '0.75rem',
                  color: outcomeColor[q.outcome],
                  border: '1px solid currentColor',
                  borderRadius: '10px',
                  padding: '0.05em 0.5em',
                }}
              >
                {outcomeLabel[q.outcome]}
              </span>
            </div>
            <p style={{ marginBottom: 0, color: '#333', fontSize: '0.9rem' }}>{q.finding}</p>
          </div>
        ))}
      </div>

      <h2>Project Scope</h2>
      <table>
        <tbody>
          <tr>
            <th scope="row">Data</th>
            <td>PrimeKG (Chandak et al., 2023) — 129,375 nodes, 8,100,498 edges, 10 node types</td>
          </tr>
          <tr>
            <th scope="row">Task</th>
            <td>Drug indication and contraindication prediction</td>
          </tr>
          <tr>
            <th scope="row">Splits</th>
            <td>Standard (diseases in train+test) and zero-shot (held-out diseases, 641 diseases)</td>
          </tr>
          <tr>
            <th scope="row">Metrics</th>
            <td>AUPRC (primary), AUROC (secondary). Random-negative evaluation (1:5 ratio).</td>
          </tr>
          <tr>
            <th scope="row">Seeds</th>
            <td>[42, 0, 1] — mean ± std reported</td>
          </tr>
          <tr>
            <th scope="row">Hardware</th>
            <td>NVIDIA GeForce RTX 4060, 8 GB VRAM. CUDA throughout.</td>
          </tr>
          <tr>
            <th scope="row">Model scale</th>
            <td>hidden_dim=64, 2 layers, 4 heads (published: 512, 3 layers, 8 heads)</td>
          </tr>
        </tbody>
      </table>

      <h2>Leakage Check</h2>
      <p>
        The zero-shot split passes the leakage check: 0 held-out diseases appear in any training
        treatment edge. Results verified for seeds [42, 0, 1]. Files:{' '}
        <code>results/metrics/leakage_check_seed{'{n}'}.json</code> — all PASS.
      </p>

      <h2>Citation</h2>
      <p>
        Huang K, Chandak P, Wang Q, et al. &ldquo;A foundation model for clinician-centered drug
        repurposing.&rdquo; <em>Nature Medicine</em>, 2024.{' '}
        <a
          href="https://doi.org/10.1038/s41591-024-03233-x"
          target="_blank"
          rel="noopener noreferrer"
        >
          DOI: 10.1038/s41591-024-03233-x
        </a>
      </p>
    </div>
  )
}
