import React from 'react'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts'
import { useData } from '../context/DataContext.jsx'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

const BIN_ORDER = ['0', '1-5', '6-20', '21+']
const BIN_LABEL = {
  '0': '0 edges',
  '1-5': '1-5 edges',
  '6-20': '6-20 edges',
  '21+': '21+ edges',
}

export default function Q3Zeroshot() {
  const { data, loading, error } = useData()

  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const { binned } = data.degradationCurveData || { binned: [] }

  // Build chart data: one row per bin, one column per model+relation combo
  const models = [...new Set(binned.map((r) => r.model))]
  const relations = [...new Set(binned.map((r) => r.relation))]

  const chartDataInd = BIN_ORDER.map((bin) => {
    const obj = { bin: BIN_LABEL[bin] || bin }
    models.forEach((m) => {
      const row = binned.find((r) => r.model === m && r.bin === bin && r.relation === 'indication')
      if (row) obj[m] = row.mean_auprc
    })
    return obj
  }).filter((row) => models.some((m) => row[m] !== undefined))

  const chartDataContra = BIN_ORDER.map((bin) => {
    const obj = { bin: BIN_LABEL[bin] || bin }
    models.forEach((m) => {
      const row = binned.find(
        (r) => r.model === m && r.bin === bin && r.relation === 'contraindication'
      )
      if (row) obj[m] = row.mean_auprc
    })
    return obj
  }).filter((row) => models.some((m) => row[m] !== undefined))

  const COLORS = ['var(--accent)', 'var(--indication)', 'var(--structure)', 'var(--contra)']

  return (
    <div className="page">
      <h1>Q3 — Why Is Zero-Shot Prediction Preferred?</h1>

      <section className="section" aria-labelledby="conceptual">
        <h2 id="conceptual">Conceptual Answer</h2>
        <p>
          Huang et al. (2024) report that 92% of PrimeKG diseases have no indication edge (PMC11326339).
          On our downloaded PrimeKG, 9,388 unique indication pairs cover 17,080 diseases — consistent
          with the ~8% coverage figure. A supervised classifier requires labeled drug-disease pairs per
          disease. For the remaining ~92% of diseases, no such labels exist.
        </p>
        <p>
          Zero-shot generalization via disease similarity is not a design preference. It is the only
          tractable path for rare disease coverage. A model trained with the standard split cannot make
          any prediction for a disease with no training treatment edges.
        </p>
        <div className="finding-box">
          <p>
            <strong>92% of PrimeKG diseases have no indication edge.</strong> Source: Huang et al.
            (2024), main text (PMC11326339). This means a standard supervised model—which requires
            labeled drug-disease training pairs—can make no prediction for 92% of diseases. Zero-shot
            generalization is the only approach that covers these diseases.
          </p>
        </div>
      </section>

      <section className="section" aria-labelledby="empirical">
        <h2 id="empirical">Empirical: Degradation Curve</h2>
        <p>
          AUPRC as a function of the number of training-time treatment edges per disease, binned
          into four groups. Each bin shows mean AUPRC across all diseases in that bin.
        </p>

        <h3>Indication AUPRC by Training Edge Count</h3>
        {chartDataInd.length > 0 ? (
          <>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartDataInd} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="bin" />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v) => v.toFixed(2)}
                    label={{ value: 'Mean AUPRC', angle: -90, position: 'insideLeft', dx: -10 }}
                  />
                  <Tooltip formatter={(v) => (v !== undefined ? v.toFixed(4) : '—')} />
                  <Legend />
                  {models.map((m, i) => (
                    <Bar key={m} dataKey={m} fill={COLORS[i % COLORS.length]} name={m} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <ProvenanceCaption
              source="results/metrics/degradation_curve_data.json"
              extra="binned by n_train_edges: [0], [1-5], [6-20], [21+]"
            />
          </>
        ) : (
          <p style={{ color: '#555' }}>
            No indication data in degradation_curve_data.json. The file contains{' '}
            {data.degradationCurveData?.raw?.length ?? 0} raw rows.
          </p>
        )}

        <h3>Contraindication AUPRC by Training Edge Count</h3>
        {chartDataContra.length > 0 ? (
          <>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartDataContra} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="bin" />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v) => v.toFixed(2)}
                    label={{ value: 'Mean AUPRC', angle: -90, position: 'insideLeft', dx: -10 }}
                  />
                  <Tooltip formatter={(v) => (v !== undefined ? v.toFixed(4) : '—')} />
                  <Legend />
                  {models.map((m, i) => (
                    <Bar key={m} dataKey={m} fill={COLORS[i % COLORS.length]} name={m} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <ProvenanceCaption
              source="results/metrics/degradation_curve_data.json"
              extra="contraindication relation, binned"
            />
          </>
        ) : (
          <p style={{ color: '#555' }}>No contraindication data available in degradation curve.</p>
        )}

        <h3>Binned Summary Table</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Relation</th>
                <th>Bin</th>
                <th>Mean AUPRC</th>
                <th>N diseases</th>
              </tr>
            </thead>
            <tbody>
              {binned.map((row, i) => (
                <tr key={i}>
                  <td>{row.model}</td>
                  <td>{row.relation}</td>
                  <td>{BIN_LABEL[row.bin] || row.bin}</td>
                  <td>{row.mean_auprc.toFixed(4)}</td>
                  <td>{row.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ProvenanceCaption source="results/metrics/degradation_curve_data.json" />
      </section>

      <section className="section" aria-labelledby="interpretation">
        <h2 id="interpretation">Interpretation</h2>
        <p>
          The degradation curve shows that models evaluated on the zero-shot split (all test
          diseases have 0 training edges) face the hardest case. TxGNN&rsquo;s disease-similarity
          module provides generalization to these diseases by propagating signals from
          structurally similar diseases that do have training edges.
        </p>
        <p>
          This analysis is specific to: PrimeKG, scaled reproduction (hidden_dim=64), seed=42,
          degradation curve data from{' '}
          <code>results/metrics/degradation_curve_data.json</code>.
        </p>
      </section>
    </div>
  )
}
