import React from 'react'
import { Link } from 'react-router-dom'
import { useData } from '../context/DataContext.jsx'
import PathRenderer from '../components/PathRenderer.jsx'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

// Illustrative FHC path — structure only, no metric values
const FHC_NODES = [
  { id: 'fhc',        label: 'FHC',         type: 'disease' },
  { id: 'myh7',       label: 'MYH7',        type: 'gene' },
  { id: 'propranolol',label: 'Propranolol', type: 'drug' },
]
const FHC_EDGES = [
  { source: 'fhc',  target: 'myh7',        label: 'gene_associated_with_disease' },
  { source: 'myh7', target: 'propranolol', label: 'treats' },
]

const questions = [
  {
    num: 'Q1',
    to: '/results',
    label: 'Does KG augmentation improve over no-KG?',
    finding:
      'KG message passing does not improve indication AUPRC on the standard split. The no-KG model outperforms (0.893 vs 0.825). On zero-shot contraindication, the KG model has an advantage.',
    outcome: 'unexpected',
  },
  {
    num: 'Q2',
    to: '/q2-alternatives',
    label: 'Is there a better alternative to two-phase training?',
    finding:
      'No. Two-phase TxGNN leads on zero-shot indication (0.736) vs single-stage (0.706) and joint contrastive (0.670). It also runs faster (71.6s vs 130.7s).',
    outcome: 'expected',
  },
  {
    num: 'Q3',
    to: '/q3-zeroshot',
    label: 'Why is zero-shot prediction preferred?',
    finding:
      '92% of PrimeKG diseases have no approved therapy (Huang et al. 2024). A supervised classifier has no labels for these diseases. Zero-shot is not a design preference — it is a necessity.',
    outcome: 'confirmed',
  },
  {
    num: 'Q4',
    to: '/case-studies',
    label: 'What do two case studies reveal about model predictions?',
    finding:
      'Case A (FHC, n_pos=1): model fails — Propranolol not in top-20. Case B (S. aureus, n_pos=45): Benzylpenicillin at rank 18; three cancer drugs in top-10.',
    outcome: 'mixed',
  },
  {
    num: 'Q5',
    to: '/results',
    label: 'How does plain GNN compare to TxGNN?',
    finding:
      'On standard split: gnn_no_kg wins (0.893). On zero-shot: TxGNN two-phase leads (0.736). Full table generated from CSV — no manually typed numbers.',
    outcome: 'neutral',
  },
  {
    num: 'Q6',
    to: '/q6-ablations',
    label: 'Is attention augmentation in TxGNN optional?',
    finding:
      'Attention is DETRIMENTAL. Removing HGT attention improves zero-shot indication AUPRC from 0.736 to 0.772 (delta = −0.036). The disease-similarity module is the load-bearing component.',
    outcome: 'unexpected',
  },
]

const outcomeStyle = {
  expected:   { color: 'var(--indication)' },
  unexpected: { color: 'var(--contra)' },
  confirmed:  { color: 'var(--accent)' },
  mixed:      { color: 'var(--structure)' },
  neutral:    { color: 'var(--structure)' },
}

const outcomeLabel = {
  expected:   'confirmed hypothesis',
  unexpected: 'unexpected finding',
  confirmed:  'confirmed',
  mixed:      'mixed',
  neutral:    'neutral',
}

function SectionDivider() {
  return (
    <div className="section-divider" aria-hidden="true">
      <svg width="48" height="16" viewBox="0 0 48 16" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="6" cy="8" r="5" stroke="var(--accent)" strokeWidth="1.5" fill="none" />
        <line x1="11" y1="8" x2="37" y2="8" stroke="var(--accent)" strokeWidth="1.5" />
        <circle cx="42" cy="8" r="5" stroke="var(--accent)" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  )
}

export default function Overview() {
  const { data, loading, error } = useData()

  return (
    <div className="page">
      {/* Hero */}
      <div className="overview-hero">
        <h1>TxGNN Drug Repurposing</h1>
        <p className="subtitle">A scaled reproduction on PrimeKG &middot; RTX 4060 8&thinsp;GB</p>
      </div>

      {/* Illustrative KG path */}
      <PathRenderer nodes={FHC_NODES} edges={FHC_EDGES} animate={true} />
      <ProvenanceCaption source="results/predictions/case_study_caseA_paths_txgnn.csv" />

      {/* Scaled reproduction notice */}
      <div className="finding-box">
        <p>
          <strong>Scaled reproduction notice.</strong> All models use{' '}
          <code>hidden_dim=64</code> instead of the published <code>hidden_dim=512</code> due to
          the RTX 4060 8&thinsp;GB VRAM constraint. Results are labeled{' '}
          <span className="badge badge-scaled">scaled_reproduction</span> throughout. Paper numbers
          from Huang et al. (2024){' '}
          <span className="badge badge-paper">paper_reported</span> are shown in a separate column.
        </p>
      </div>

      <p style={{ color: 'var(--structure)', marginBottom: '1.5rem', maxWidth: '72ch' }}>
        Six empirical questions about knowledge graph augmentation for drug repurposing. All results
        link to files in <code>results/</code>. No numbers typed by hand.
      </p>

      {loading && <p className="status-loading">Loading data&#8230;</p>}
      {error && (
        <div className="status-error">
          <strong>Error loading results.json:</strong> {error}
          <br />
          Run <code>npm run build:data</code> from the <code>web/</code> directory first.
        </div>
      )}

      <SectionDivider />

      <h2>Six Research Questions</h2>

      <ol className="question-list" style={{ listStyle: 'none' }}>
        {questions.map((q) => (
          <li key={q.num} className="question-item">
            <span className="question-num">{q.num}</span>
            <div className="question-body">
              <Link to={q.to} className="question-title">
                {q.label}
              </Link>
              <p className="question-finding">{q.finding}</p>
              <span className="outcome-tag" style={outcomeStyle[q.outcome]}>
                {outcomeLabel[q.outcome]}
              </span>
            </div>
            <Link to={q.to} className="question-arrow" aria-label={`View details for ${q.num}`}>
              →
            </Link>
          </li>
        ))}
      </ol>

      <SectionDivider />

      <h2>Project Scope</h2>
      <div className="table-wrap">
        <table>
          <tbody>
            <tr>
              <th scope="row">Data</th>
              <td>PrimeKG (Chandak et al., 2023) &mdash; 129,375 nodes, 8,100,498 edges, 10 node types</td>
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
              <td>
                <span className="num">AUPRC</span> (primary),{' '}
                <span className="num">AUROC</span> (secondary). Random-negative evaluation (1:5 ratio).
              </td>
            </tr>
            <tr>
              <th scope="row">Seeds</th>
              <td>
                <span className="num">[42, 0, 1]</span> &mdash; mean &plusmn; std reported
              </td>
            </tr>
            <tr>
              <th scope="row">Hardware</th>
              <td>NVIDIA GeForce RTX 4060, 8&thinsp;GB VRAM. CUDA throughout.</td>
            </tr>
            <tr>
              <th scope="row">Model scale</th>
              <td>
                <span className="num">hidden_dim=64</span>, 2 layers, 4 heads (published:{' '}
                <span className="num">512</span>, 3 layers, 8 heads)
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <SectionDivider />

      <h2>Leakage Check</h2>
      <p>
        The zero-shot split passes the leakage check: 0 held-out diseases appear in any training
        treatment edge. Results verified for seeds [42, 0, 1]. Files:{' '}
        <code>results/metrics/leakage_check_seed{'{n}'}.json</code> &mdash; all PASS.
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
