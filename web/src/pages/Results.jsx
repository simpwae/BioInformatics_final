import React, { useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { useData } from '../context/DataContext.jsx'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

function fmt(val, std) {
  if (val === null || val === undefined) return '—'
  const s = val.toFixed(3)
  if (std !== null && std !== undefined) return `${s} ± ${std.toFixed(3)}`
  return s
}

function MonoNum({ children }) {
  return (
    <span style={{ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>
      {children}
    </span>
  )
}

function Q1Chart({ rows }) {
  const splits = ['standard', 'zeroshot']
  const models = ['gnn_no_kg', 'gnn_kg']

  const chartData = splits.map((split) => {
    const obj = { split: split === 'zeroshot' ? 'Zero-Shot' : 'Standard' }
    models.forEach((m) => {
      const row = rows.find((r) => r.model === m && r.split === split)
      if (row) {
        obj[`${m}_ind`]    = row.auprc_ind
        obj[`${m}_contra`] = row.auprc_contra
      }
    })
    return obj
  })

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
          <XAxis dataKey="split" tick={{ fontFamily: 'var(--font-mono)', fontSize: 12 }} />
          <YAxis
            domain={[0.6, 1]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
          />
          <Tooltip
            formatter={(v) => v.toFixed(3)}
            contentStyle={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              background: 'var(--paper-raised)',
            }}
          />
          <Legend wrapperStyle={{ fontFamily: 'var(--font-body)', fontSize: '0.8rem' }} />
          <Bar dataKey="gnn_no_kg_ind"    name="GNN no-KG (indication)"  fill="var(--indication)" opacity={0.85} />
          <Bar dataKey="gnn_kg_ind"       name="GNN KG (indication)"     fill="var(--accent)"     opacity={0.85} />
          <Bar dataKey="gnn_no_kg_contra" name="GNN no-KG (contra)"      fill="var(--indication)" opacity={0.45} />
          <Bar dataKey="gnn_kg_contra"    name="GNN KG (contra)"         fill="var(--accent)"     opacity={0.45} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function TableLegend() {
  return (
    <div className="table-legend">
      <div className="table-legend-item">
        <div className="legend-swatch indication" />
        <span>Indication</span>
      </div>
      <div className="table-legend-item">
        <div className="legend-swatch contra" />
        <span>Contraindication</span>
      </div>
    </div>
  )
}

function PaperCell({ pr, model, split }) {
  if (pr?.auprc_ind) {
    return (
      <span>
        <MonoNum>{pr.auprc_ind}</MonoNum>{' '}
        <span className="badge badge-paper">paper</span>
      </span>
    )
  }
  if (split === 'zeroshot' && (model === 'txgnn_two_phase' || model === 'txgnn_attn_on')) {
    return (
      <span
        className="paper-missing"
        title="Absolute zero-shot values available in Huang et al. 2024 Supplementary Tables S1–S2 (MOESM1 ESM). Relative improvement: +19.0% indication vs next-best baseline."
      >
        — <span style={{ fontSize: '0.7rem' }}>[no data — see Suppl. S1]</span>
      </span>
    )
  }
  return <span style={{ color: 'var(--structure)', fontFamily: 'var(--font-mono)' }}>—</span>
}

function PaperContraCell({ pr, model, split }) {
  if (pr?.auprc_contra) {
    return (
      <span>
        <MonoNum>{pr.auprc_contra}</MonoNum>{' '}
        <span className="badge badge-paper">paper</span>
      </span>
    )
  }
  if (split === 'zeroshot' && (model === 'txgnn_two_phase' || model === 'txgnn_attn_on')) {
    return (
      <span
        className="paper-missing"
        title="Absolute zero-shot values available in Huang et al. 2024 Supplementary Tables S1–S2 (MOESM1 ESM). Relative improvement: +23.9% contraindication vs next-best baseline."
      >
        — <span style={{ fontSize: '0.7rem' }}>[no data — see Suppl. S2]</span>
      </span>
    )
  }
  return <span style={{ color: 'var(--structure)', fontFamily: 'var(--font-mono)' }}>—</span>
}

export default function Results() {
  const { data, loading, error } = useData()
  const [splitFilter, setSplitFilter] = useState('all')

  if (loading) return <div className="page status-loading">Loading&#8230;</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const table = data.comparisonTable || []

  const q1Models = ['gnn_no_kg', 'gnn_kg']
  const q1Rows = table.filter((r) => q1Models.includes(r.model))

  const filteredTable =
    splitFilter === 'all' ? table : table.filter((r) => r.split === splitFilter)

  const splits = ['standard', 'zeroshot']
  const paperNote = data.paperReported?.zeroshot?._relativeNote || ''

  return (
    <div className="page">
      <h1>Q1 + Q5: Results</h1>

      {/* Q1 */}
      <section className="section" aria-labelledby="q1-heading">
        <h2 id="q1-heading">Q1 — Does KG Augmentation Improve Over No-KG?</h2>
        <p>
          Same HGT backbone, two conditions: (A) 2-layer HGT with KG message passing, (B) 0-layer
          (no message passing). Same Phase 1 link-prediction pretrain for both. Primary metric:
          indication AUPRC.
        </p>

        <div className="finding-box">
          <p>
            <strong>Finding:</strong> KG message passing does not improve indication AUPRC on
            either split in this scaled reproduction. On the standard split, the no-KG model
            outperforms the KG model (<MonoNum>0.893</MonoNum> vs <MonoNum>0.825</MonoNum>). On
            the zero-shot split, results are within noise for indication (
            <MonoNum>0.704</MonoNum> vs <MonoNum>0.714</MonoNum>); the KG model is better for
            contraindication (<MonoNum>0.831</MonoNum> vs <MonoNum>0.725</MonoNum>). This result
            is reported as-is per the ground rules.
          </p>
        </div>

        <Q1Chart rows={q1Rows} />
        <ProvenanceCaption
          source="results/gnn/{standard,zeroshot}/seed_{42,0,1}/gnn_baseline.json, gnn_no_kg.json"
          extra="chart values are mean AUPRC across seeds [42, 0, 1]"
        />

        <h3>Detailed Q1 Table</h3>
        <TableLegend />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Split</th>
                <th className="col-indication">AUPRC ind (mean±std)</th>
                <th className="col-contra">AUPRC contra (mean±std)</th>
                <th className="col-indication">AUROC ind</th>
                <th className="col-contra">AUROC contra</th>
                <th>Wall-clock (s)</th>
              </tr>
            </thead>
            <tbody>
              {q1Rows.map((r, i) => (
                <tr key={i}>
                  <td>{r.displayName}</td>
                  <td className={r.split === 'zeroshot' ? 'tag-zeroshot' : 'tag-standard'}>
                    {r.split === 'zeroshot' ? 'zero-shot' : 'standard'}
                  </td>
                  <td><MonoNum>{fmt(r.auprc_ind, r.auprc_ind_std)}</MonoNum></td>
                  <td><MonoNum>{fmt(r.auprc_contra, r.auprc_contra_std)}</MonoNum></td>
                  <td><MonoNum>{fmt(r.auroc_ind, r.auroc_ind_std)}</MonoNum></td>
                  <td><MonoNum>{fmt(r.auroc_contra, r.auroc_contra_std)}</MonoNum></td>
                  <td><MonoNum>{r.wall_s !== null ? r.wall_s : '—'}</MonoNum></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ProvenanceCaption source="results/metrics/comparison_table.csv" />
      </section>

      {/* Q5 */}
      <section className="section" aria-labelledby="q5-heading">
        <h2 id="q5-heading">Q5 — Full Comparison Table: GNN vs TxGNN</h2>
        <p>
          Auto-generated from <code>results/metrics/comparison_table.csv</code>. All models,
          both splits, seeds [42, 0, 1]. Random-negative AUPRC (1:5 ratio). The{' '}
          <span className="badge badge-paper">paper_reported</span> column shows values from
          Huang et al. (2024), Nature Medicine, Supplementary Tables S1&ndash;S2. Never merged
          with our numbers.
        </p>

        {paperNote && (
          <div className="warn-box">
            <p>
              <strong>Zero-shot paper_reported note:</strong> {paperNote}
            </p>
          </div>
        )}

        <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ lineHeight: '2', marginRight: '0.5rem', fontSize: '0.875rem', color: 'var(--structure)' }}>
            Filter split:
          </span>
          {['all', 'standard', 'zeroshot'].map((s) => (
            <button
              key={s}
              onClick={() => setSplitFilter(s)}
              aria-pressed={splitFilter === s}
              style={{
                padding: '0.3rem 0.75rem',
                border: '1px solid var(--accent)',
                borderRadius: 'var(--radius)',
                cursor: 'pointer',
                background: splitFilter === s ? 'var(--accent)' : 'var(--paper-raised)',
                color: splitFilter === s ? '#fff' : 'var(--accent)',
                fontSize: '0.8125rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: splitFilter === s ? 600 : 400,
              }}
            >
              {s === 'all' ? 'All' : s === 'zeroshot' ? 'Zero-Shot' : 'Standard'}
            </button>
          ))}
        </div>

        {splits
          .filter((s) => splitFilter === 'all' || splitFilter === s)
          .map((split) => {
            const rows = filteredTable.filter((r) => r.split === split)
            return (
              <div key={split}>
                <h3>{split === 'zeroshot' ? 'Zero-Shot Split' : 'Standard Split'}</h3>
                <TableLegend />
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Type</th>
                        <th className="col-indication">AUPRC ind (scaled)</th>
                        <th className="col-indication">AUPRC ind (paper)</th>
                        <th className="col-contra">AUPRC contra (scaled)</th>
                        <th className="col-contra">AUPRC contra (paper)</th>
                        <th>Wall-clock (s)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r, i) => {
                        const pr = r.paper_reported
                        return (
                          <tr key={i}>
                            <td>{r.displayName}</td>
                            <td>
                              <span
                                className={
                                  r.reproduction_type === 'scaled_reproduction'
                                    ? 'badge badge-scaled'
                                    : r.reproduction_type === 'original_ablation'
                                    ? 'badge badge-original'
                                    : 'badge'
                                }
                              >
                                {r.reproduction_type}
                              </span>
                            </td>
                            <td><MonoNum>{fmt(r.auprc_ind, r.auprc_ind_std)}</MonoNum></td>
                            <td>
                              <PaperCell pr={pr} model={r.model} split={split} />
                            </td>
                            <td><MonoNum>{fmt(r.auprc_contra, r.auprc_contra_std)}</MonoNum></td>
                            <td>
                              <PaperContraCell pr={pr} model={r.model} split={split} />
                            </td>
                            <td><MonoNum>{r.wall_s !== null ? r.wall_s : '—'}</MonoNum></td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })}

        <ProvenanceCaption
          source="results/metrics/comparison_table.csv"
          extra="paper_reported: Huang et al. 2024, Nat Med, Suppl. Tables S1-S2 (MOESM1 ESM)"
        />

        <p style={{ fontSize: '0.85rem', color: 'var(--structure)', fontFamily: 'var(--font-mono)' }}>
          Note: <code>txgnn_two_phase</code> and <code>txgnn_attn_on</code> are the same model —
          both rows are included as the CSV contains both. The table includes only models where all
          results are available (rows with &ldquo;[NOT YET RUN]&rdquo; are excluded).
        </p>
      </section>
    </div>
  )
}
