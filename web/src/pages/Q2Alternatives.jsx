import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { useData } from '../context/DataContext.jsx'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

function fmt(val, std) {
  if (val === null || val === undefined) return '—'
  if (std !== null && std !== undefined) return `${val.toFixed(3)} ± ${std.toFixed(3)}`
  return val.toFixed(3)
}

const Q2_MODELS = ['txgnn_two_phase', 'single_stage', 'joint_contrastive']

export default function Q2Alternatives() {
  const { data, loading, error } = useData()

  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const table = data.comparisonTable || []
  const q2Rows = table.filter((r) => Q2_MODELS.includes(r.model))

  const zeroRows = q2Rows.filter((r) => r.split === 'zeroshot')
  const stdRows = q2Rows.filter((r) => r.split === 'standard')

  const chartDataZero = zeroRows.map((r) => ({
    name: r.displayName,
    'AUPRC ind': r.auprc_ind,
    'AUPRC contra': r.auprc_contra,
    'Wall-clock (s)': r.wall_s,
  }))

  const chartDataStd = stdRows.map((r) => ({
    name: r.displayName,
    'AUPRC ind': r.auprc_ind,
    'AUPRC contra': r.auprc_contra,
  }))

  return (
    <div className="page">
      <h1>Q2 — Alternatives to Two-Phase Training</h1>

      <p>
        Baseline: TxGNN two-phase (Phase 1 = self-supervised KG pretraining; Phase 2 = therapeutic
        task fine-tuning). Two alternatives tested on the same backbone (hidden_dim=64, 2 layers, 4
        heads).
      </p>

      <div className="finding-box">
        <p>
          <strong>Pre-defined criterion (set before running):</strong> higher AUPRC on zero-shot
          indication split at equal or lower total wall-clock time.
        </p>
        <p>
          <strong>Finding:</strong> Two-phase training outperforms both alternatives on zero-shot
          indication (0.736 vs 0.706 vs 0.670) and runs faster than both (71.6s vs 130.7s vs
          97.1s). No alternative beats the baseline under the pre-defined criterion.
        </p>
      </div>

      <h2>Zero-Shot Split Results</h2>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartDataZero} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0.6, 0.9]} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip formatter={(v) => v.toFixed(3)} />
            <Legend />
            <Bar dataKey="AUPRC ind" fill="#333333" />
            <Bar dataKey="AUPRC contra" fill="#888888" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ProvenanceCaption source="results/metrics/comparison_table.csv" extra="zero-shot split, seeds [42, 0, 1]" />

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>AUPRC ind (mean±std)</th>
              <th>AUPRC contra (mean±std)</th>
              <th>AUROC ind</th>
              <th>Wall-clock (s)</th>
            </tr>
          </thead>
          <tbody>
            {zeroRows.map((r, i) => (
              <tr key={i} style={r.model === 'txgnn_two_phase' ? { fontWeight: 600 } : {}}>
                <td>
                  {r.displayName}
                  {r.model === 'txgnn_two_phase' && (
                    <span style={{ marginLeft: '0.5rem', fontWeight: 400, fontSize: '0.8rem', color: '#1a5c2e' }}>
                      (best)
                    </span>
                  )}
                </td>
                <td>{fmt(r.auprc_ind, r.auprc_ind_std)}</td>
                <td>{fmt(r.auprc_contra, r.auprc_contra_std)}</td>
                <td>{fmt(r.auroc_ind, r.auroc_ind_std)}</td>
                <td>{r.wall_s !== null ? r.wall_s : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source="results/metrics/comparison_table.csv" />

      <h2>Standard Split Results</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>AUPRC ind (mean±std)</th>
              <th>AUPRC contra (mean±std)</th>
              <th>Wall-clock (s)</th>
            </tr>
          </thead>
          <tbody>
            {stdRows.map((r, i) => (
              <tr key={i}>
                <td>{r.displayName}</td>
                <td>{fmt(r.auprc_ind, r.auprc_ind_std)}</td>
                <td>{fmt(r.auprc_contra, r.auprc_contra_std)}</td>
                <td>{r.wall_s !== null ? r.wall_s : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source="results/metrics/comparison_table.csv" />

      <h2>Interpretation</h2>
      <p>
        The pre-training phase (Phase 1: KG link prediction) positions embeddings in a space where
        the subsequent fine-tuning on therapeutic tasks generalizes better to unseen diseases.
        Collapsing the two phases loses this benefit.
      </p>
      <p>
        The joint contrastive approach shows higher zero-shot contraindication AUPRC (0.854) but
        lower and more variable indication AUPRC (0.670 ± 0.038). This trade-off is noted, but
        under the pre-defined criterion (indication AUPRC), it does not beat two-phase training.
      </p>

      <h2>Trade-off Table</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Zero-shot ind AUPRC</th>
              <th>Wall-clock (s)</th>
              <th>Beats two-phase (ind)?</th>
              <th>Beats two-phase (speed)?</th>
            </tr>
          </thead>
          <tbody>
            {zeroRows.map((r, i) => {
              const twoPhase = zeroRows.find((x) => x.model === 'txgnn_two_phase')
              const betterInd = twoPhase ? r.auprc_ind > twoPhase.auprc_ind : false
              const betterSpeed = twoPhase ? r.wall_s < twoPhase.wall_s : false
              return (
                <tr key={i}>
                  <td>{r.displayName}</td>
                  <td>{r.auprc_ind?.toFixed(3)}</td>
                  <td>{r.wall_s}</td>
                  <td
                    style={{
                      color: betterInd ? '#1a5c2e' : '#7a1818',
                      fontWeight: 600,
                    }}
                  >
                    {r.model === 'txgnn_two_phase' ? '(baseline)' : betterInd ? 'Yes' : 'No'}
                  </td>
                  <td
                    style={{
                      color: betterSpeed ? '#1a5c2e' : '#7a1818',
                      fontWeight: 600,
                    }}
                  >
                    {r.model === 'txgnn_two_phase' ? '(baseline)' : betterSpeed ? 'Yes' : 'No'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source="results/metrics/comparison_table.csv" extra="zero-shot split" />
    </div>
  )
}
