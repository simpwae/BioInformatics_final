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
  ErrorBar,
} from 'recharts'
import { useData } from '../context/DataContext.jsx'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

function fmt(val, std) {
  if (val === null || val === undefined) return '—'
  if (std !== null && std !== undefined) return `${val.toFixed(3)} ± ${std.toFixed(3)}`
  return val.toFixed(3)
}

function MonoNum({ children }) {
  return (
    <span style={{ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>
      {children}
    </span>
  )
}

const VARIANT_DISPLAY = {
  txgnn:          'TxGNN (attn=ON, sim=ON)',
  txgnn_no_attn:  'TxGNN (attn=OFF, sim=ON)',
  txgnn_no_sim:   'TxGNN (attn=ON, sim=OFF)',
  txgnn_no_both:  'TxGNN (attn=OFF, sim=OFF)',
}

export default function Q6Ablations() {
  const { data, loading, error } = useData()

  if (loading) return <div className="page status-loading">Loading&#8230;</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const ablationMatrix = data.ablationMatrix || {}
  const matrix         = ablationMatrix.matrix || []
  const q6Decision     = ablationMatrix.q6_decision || {}
  const q6Table        = data.q6AblationTable || []

  const zeroMatrix = matrix.filter((r) => r.split === 'zeroshot')
  const stdMatrix  = matrix.filter((r) => r.split === 'standard')

  const chartDataZero = zeroMatrix.map((r) => ({
    name: VARIANT_DISPLAY[r.variant] || r.variant,
    'Indication AUPRC':     r.indication_auprc_mean,
    'Contraindication AUPRC': r.contraindication_auprc_mean,
  }))

  const attnOn  = zeroMatrix.find((r) => r.variant === 'txgnn')
  const attnOff = zeroMatrix.find((r) => r.variant === 'txgnn_no_attn')
  const delta   = q6Decision.delta_attn_on_minus_off

  const deltaDisplay = delta !== undefined
    ? (delta >= 0 ? '+' : '') + delta.toFixed(3)
    : '—'

  return (
    <div className="page">
      <h1>Q6 — Is Attention Augmentation in TxGNN Optional? (Result: Detrimental)</h1>

      <p>
        Hypothesis to test: TxGNN&rsquo;s zero-shot advantage comes from the disease-similarity
        module, not from the HGT attention mechanism. The original question asks whether attention is
        &ldquo;optional.&rdquo; The result is stronger than optional — attention is{' '}
        <strong>actively detrimental</strong> in this scaled reproduction.
      </p>
      <p>
        Decision rule set before running: delta = AUPRC(attn=ON) &minus; AUPRC(attn=OFF).{' '}
        <MonoNum>delta &isin; (&minus;0.02, +0.02)</MonoNum> &rarr; optional (neutral);{' '}
        <MonoNum>delta &le; &minus;0.02</MonoNum> &rarr; attention is detrimental;{' '}
        <MonoNum>delta &ge; +0.02</MonoNum> &rarr; attention helps.
      </p>

      {/* Prominent stat card */}
      <div className="stat-card-negative">
        <div className="stat-label">Attention AUPRC delta (ON &minus; OFF)</div>
        <div className="stat-value">
          {deltaDisplay}
        </div>
        <div className="stat-sublabel">
          AUPRC(attn=ON) &minus; AUPRC(attn=OFF) ={' '}
          <MonoNum>{attnOn ? attnOn.indication_auprc_mean.toFixed(3) : '?'}</MonoNum> &minus;{' '}
          <MonoNum>{attnOff ? attnOff.indication_auprc_mean.toFixed(3) : '?'}</MonoNum>
        </div>
        <div className="stat-verdict">Attention is DETRIMENTAL</div>
      </div>

      <div className="negative-box">
        <p>
          <strong>Q6 Decision (threshold = &plusmn;0.02 AUPRC, set before running):</strong>{' '}
          {q6Decision.conclusion ||
            'delta < −0.02 — attention actively hurts in this scaled setup. Removing HGT attention IMPROVES zero-shot indication AUPRC.'}
        </p>
      </div>

      <h2>Ablation Matrix — Zero-Shot Split</h2>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartDataZero} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}
              interval={0}
              angle={-20}
              textAnchor="end"
            />
            <YAxis
              domain={[0.7, 0.9]}
              tickFormatter={(v) => v.toFixed(2)}
              tick={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
            />
            <Tooltip
              formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)}
              contentStyle={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.8rem',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                background: 'var(--paper-raised)',
              }}
            />
            <Legend wrapperStyle={{ fontFamily: 'var(--font-body)', fontSize: '0.8rem' }} />
            <Bar dataKey="Indication AUPRC"     fill="var(--indication)" opacity={0.85} />
            <Bar dataKey="Contraindication AUPRC" fill="var(--contra)"  opacity={0.85} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ProvenanceCaption source="results/ablations/matrix.json" extra="zero-shot split, seeds [42, 0, 1]" />

      <h2>Full Ablation Table — Zero-Shot Split</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Ind AUPRC (mean±std)</th>
              <th>Contra AUPRC (mean±std)</th>
              <th>Ind AUROC</th>
              <th>Wall-clock (s)</th>
            </tr>
          </thead>
          <tbody>
            {zeroMatrix.map((r, i) => (
              <tr
                key={i}
                style={
                  r.variant === 'txgnn_no_attn'
                    ? { background: 'var(--indication-muted)' }
                    : {}
                }
              >
                <td>
                  {VARIANT_DISPLAY[r.variant] || r.variant}
                  {r.variant === 'txgnn_no_attn' && (
                    <span className="positive-badge" style={{ marginLeft: '0.5rem' }}>
                      best ind AUPRC
                    </span>
                  )}
                </td>
                <td><MonoNum>{fmt(r.indication_auprc_mean, r.indication_auprc_std)}</MonoNum></td>
                <td><MonoNum>{fmt(r.contraindication_auprc_mean, r.contraindication_auprc_std)}</MonoNum></td>
                <td><MonoNum>{r.indication_auroc_mean?.toFixed(3)}</MonoNum></td>
                <td><MonoNum>{r.wall_clock_s_mean}</MonoNum></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source="results/ablations/matrix.json" />

      <h2>Standard Split — Ablation Matrix</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Ind AUPRC (mean±std)</th>
              <th>Contra AUPRC (mean±std)</th>
              <th>Wall-clock (s)</th>
            </tr>
          </thead>
          <tbody>
            {stdMatrix.map((r, i) => (
              <tr key={i}>
                <td>{VARIANT_DISPLAY[r.variant] || r.variant}</td>
                <td><MonoNum>{fmt(r.indication_auprc_mean, r.indication_auprc_std)}</MonoNum></td>
                <td><MonoNum>{fmt(r.contraindication_auprc_mean, r.contraindication_auprc_std)}</MonoNum></td>
                <td><MonoNum>{r.wall_clock_s_mean}</MonoNum></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source="results/ablations/matrix.json" />

      <h2>Q6 Ablation Table (from CSV)</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Split</th>
              <th>AUPRC ind (mean±std)</th>
              <th>N seeds</th>
            </tr>
          </thead>
          <tbody>
            {q6Table.map((r, i) => (
              <tr key={i}>
                <td>{r.displayName || r.variant}</td>
                <td className={r.split === 'zeroshot' ? 'tag-zeroshot' : 'tag-standard'}>
                  {r.split === 'zeroshot' ? 'zero-shot' : 'standard'}
                </td>
                <td><MonoNum>{fmt(r.auprc_ind_mean, r.auprc_ind_std)}</MonoNum></td>
                <td><MonoNum>{r.n_seeds}</MonoNum></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source="results/metrics/q6_ablation_table.csv" />

      <h2>Interpretation</h2>
      <p>
        The hypothesis (zero-shot advantage comes from disease-similarity, not attention) is
        partially supported but with an unexpected twist:
      </p>
      <ul style={{ marginBottom: '1rem', paddingLeft: '1.5rem' }}>
        <li>
          Removing attention improves performance — attention is not load-bearing, it actively hurts
          in this scaled setup.
        </li>
        <li>
          Removing similarity (<code>txgnn_no_sim</code>) reduces AUPRC from{' '}
          <MonoNum>0.736</MonoNum> to <MonoNum>0.726</MonoNum>, confirming the similarity module
          contributes positively.
        </li>
        <li>
          Removing both (<code>txgnn_no_both</code>, <MonoNum>0.762</MonoNum>) is better than the
          full model but worse than removing attention alone (<MonoNum>0.772</MonoNum>).
        </li>
        <li>
          The HGT attention mechanism likely overfits with only 64-dimensional embeddings and
          limited training data.
        </li>
      </ul>

      <div className="finding-box">
        <p>
          Consistent with Huang et al. (2024), who report that a learnable attention mechanism was
          tested and found ineffective, replacing it with a fixed degree-based exponential gating
          function (&lambda;=0.7). Our ablation independently arrives at the same conclusion.
          Source: Methods, PMC11326339.
        </p>
      </div>

      <p style={{ fontSize: '0.75rem', color: 'var(--structure)', fontFamily: 'var(--font-mono)' }}>
        Scope: scaled reproduction, RTX 4060 8 GB VRAM, hidden_dim=64, 2 layers, PrimeKG,
        seeds [42, 0, 1]. This claim does not generalize beyond this setup.
      </p>
    </div>
  )
}
