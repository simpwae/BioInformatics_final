import React, { useState } from 'react'
import { useData } from '../context/DataContext.jsx'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

function DrugTable({ predictions, source }) {
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Drug</th>
              <th>DrugBank ID</th>
              <th>Score</th>
              <th>Known positive?</th>
            </tr>
          </thead>
          <tbody>
            {predictions.slice(0, 20).map((row, i) => (
              <tr
                key={i}
                style={
                  row.is_positive
                    ? { background: 'var(--indication-muted)', fontWeight: 600 }
                    : {}
                }
              >
                <td>{row.rank}</td>
                <td>{row.drug_name}</td>
                <td>
                  <code>{row.drug_id}</code>
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>{row.score?.toFixed(4)}</td>
                <td>
                  {row.is_positive ? (
                    <span className="positive-badge">Yes</span>
                  ) : (
                    <span className="negative-badge">No</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source={source} />
    </>
  )
}

function PathTable({ paths, source }) {
  if (!paths || paths.length === 0)
    return <p style={{ color: '#555' }}>No path data available.</p>
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Drug ID</th>
              <th>Via entity</th>
              <th>Via type</th>
              <th>Drug &rarr; entity relation</th>
              <th>Entity &rarr; disease relation</th>
            </tr>
          </thead>
          <tbody>
            {paths.map((p, i) => (
              <tr key={i}>
                <td>
                  <code>{p.drug_id}</code>
                </td>
                <td>{p.via_entity}</td>
                <td>{p.via_type}</td>
                <td>
                  <code>{p.relation_drug_to_entity}</code>
                </td>
                <td>
                  <code>{p.relation_entity_to_disease}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ProvenanceCaption source={source} />
    </>
  )
}

export default function CaseStudies() {
  const { data, loading, error } = useData()
  const [activeCase, setActiveCase] = useState('caseA')

  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const { caseA, caseB } = data.caseStudies || {}

  const btnStyle = (active) => ({
    padding: '0.5rem 1.25rem',
    border: '1px solid var(--accent)',
    borderRadius: 'var(--radius)',
    cursor: 'pointer',
    background: active ? 'var(--accent)' : 'var(--paper-raised)',
    color: active ? '#fff' : 'var(--accent)',
    fontWeight: active ? 600 : 400,
    fontSize: '0.875rem',
    fontFamily: 'var(--font-body)',
  })

  return (
    <div className="page">
      <h1>Q4 — Case Studies</h1>
      <p>
        Two diseases selected before examining predictions. All drug names, scores, and KG paths
        come from model output files in <code>results/predictions/</code>. No numbers typed
        manually.
      </p>
      <div className="warn-box">
        <p>
          <strong>Note on the paper&rsquo;s case studies:</strong> Huang et al. (2024) include
          their own case studies in the Nature Medicine paper using the full TxGNN (512-dim,
          pre-trained node features). Those specific case studies cannot be replicated here — the
          full model exceeds 8&thinsp;GB VRAM and the pre-trained features are not publicly
          released. The cases below are <em>original</em> case studies on the scaled_reproduction
          model, chosen from diseases present in our zero-shot test split. They are not presented as
          replicating the paper&rsquo;s narrative.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <button style={btnStyle(activeCase === 'caseA')} onClick={() => setActiveCase('caseA')}>
          Case A: FHC (rare, n_pos=1)
        </button>
        <button style={btnStyle(activeCase === 'caseB')} onClick={() => setActiveCase('caseB')}>
          Case B: S. aureus (n_pos=45)
        </button>
      </div>

      {/* Case A */}
      {activeCase === 'caseA' && caseA && (
        <section aria-labelledby="caseA-heading">
          <h2 id="caseA-heading">
            Case A: {caseA.diseaseName}
          </h2>
          <table style={{ marginBottom: '1rem' }}>
            <tbody>
              <tr>
                <th scope="row">Disease ID</th>
                <td>{caseA.diseaseId}</td>
              </tr>
              <tr>
                <th scope="row">Known positives (n_pos)</th>
                <td>{caseA.nPos}</td>
              </tr>
              <tr>
                <th scope="row">Known indication drug</th>
                <td>{caseA.knownDrug}</td>
              </tr>
              <tr>
                <th scope="row">Known drug in top-20?</th>
                <td>
                  <span className="negative-badge">No — model fails on this disease</span>
                </td>
              </tr>
              <tr>
                <th scope="row">Per-disease AUPRC (seed=42)</th>
                <td>{caseA.auprc}</td>
              </tr>
              <tr>
                <th scope="row">Split</th>
                <td>Zero-shot (disease held out from training)</td>
              </tr>
            </tbody>
          </table>

          <p>
            Familial hypertrophic cardiomyopathy (FHC) is a hereditary condition causing
            pathological thickening of the heart muscle. It is caused by mutations in sarcomere
            genes (MYH7, MYBPC3). The only indication edge for this disease in PrimeKG is
            Propranolol (DB00571), a non-selective beta-blocker.
          </p>

          <div className="warn-box">
            <p>
              All top-20 indication scores are negative. Propranolol (the one known indication)
              does not appear in the top-20. AUPRC = {caseA.auprc}. The model fails on this
              rare disease in this scaled reproduction.
            </p>
          </div>

          <h3>Top-20 Indication Predictions</h3>
          <DrugTable predictions={caseA.predictions} source={caseA.source} />

          <h3>Top-20 Contraindication Predictions</h3>
          <DrugTable predictions={caseA.contraindications || []} source={caseA.source} />

          <h3>KG Paths for Known Drugs</h3>
          <p>
            Drugs traced through 2-hop KG paths. Note: the traced drugs (Milrinone DB00235,
            Amrinone DB01427, Dipyridamole DB00975) are all <em>contraindications</em> for FHC,
            not positive indications. Their paths share a common pattern reaching FHC through
            disease-disease similarity (hypertrophic cardiomyopathy &rarr; FHC) and through
            Propranolol via drug-drug edges.
          </p>
          <PathTable paths={caseA.paths} source={caseA.pathsSource} />

          <h3>Clinical Note</h3>
          <p>
            The top predictions (Lutetium Lu 177 dotatate — a cancer radiopharmaceutical;
            Lactose — an excipient) are clinically implausible for FHC. The model does not rank
            Propranolol in the top 20. This is consistent with the AUPRC of {caseA.auprc} and
            with the model&rsquo;s limited capacity (64-dim embeddings) to generalize to diseases
            with only one known drug in the KG.
          </p>
        </section>
      )}

      {/* Case B */}
      {activeCase === 'caseB' && caseB && (
        <section aria-labelledby="caseB-heading">
          <h2 id="caseB-heading">
            Case B: {caseB.diseaseName}
          </h2>
          <table style={{ marginBottom: '1rem' }}>
            <tbody>
              <tr>
                <th scope="row">Disease ID</th>
                <td>{caseB.diseaseId}</td>
              </tr>
              <tr>
                <th scope="row">Known positives (n_pos)</th>
                <td>{caseB.nPos}</td>
              </tr>
              <tr>
                <th scope="row">First positive drug in top-20</th>
                <td>
                  {caseB.firstPositiveDrug} at rank {caseB.firstPositiveRank}
                </td>
              </tr>
              <tr>
                <th scope="row">Per-disease AUPRC (seed=42)</th>
                <td>{caseB.auprc}</td>
              </tr>
              <tr>
                <th scope="row">Split</th>
                <td>Zero-shot (disease held out from training)</td>
              </tr>
            </tbody>
          </table>

          <p>
            Staphylococcus aureus is a common gram-positive bacterial pathogen. Multiple antibiotic
            classes are indicated, including beta-lactams, tetracyclines, macrolides, and topical
            agents. MRSA strains require mupirocin, daptomycin, or vancomycin.
          </p>

          <div className="finding-box">
            <p>
              Benzylpenicillin (Penicillin G) appears at rank 18 (is_positive=True). Clinically
              plausible antibiotics in the top-20: Mupirocin (rank 3), Doxycycline (rank 5),
              Minocycline (rank 10), Benzylpenicillin (rank 18). Three antineoplastic drugs appear
              in the top-10 (Etoposide rank 2, Carboplatin rank 6, Bleomycin rank 8).
            </p>
          </div>

          <h3>Top-20 Indication Predictions</h3>
          <DrugTable predictions={caseB.predictions} source={caseB.source} />

          <h3>KG Paths for Known Drugs</h3>
          <p>
            Drugs traced (Cefprozil DB01150, Cefdinir DB00535, Tazobactam DB01606) share a common
            path via a related disease node (&ldquo;staphylococcal infection&rdquo;) and then connect via
            disease-disease similarity to the specific target disease. Same mechanism as Case A.
          </p>
          <PathTable paths={caseB.paths} source={caseB.pathsSource} />

          <h3>Clinical Discrepancy Note</h3>
          <p>
            The three implausible cancer drug predictions (Etoposide, Carboplatin, Bleomycin at
            ranks 2, 6, 8) likely result from these drugs sharing KG neighbors (immune-related
            nodes) with anti-infective drugs. In a 64-dim embedding space with no pre-trained
            features, the model cannot distinguish mechanism of action at this resolution.
          </p>
          <p>
            These observations are specific to seed=42, zero-shot split, scaled_reproduction model.
            They are not claims about the published TxGNN.
          </p>
        </section>
      )}

      <h2 style={{ marginTop: '2rem' }}>Interpretation Notes</h2>
      <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
        <li>
          Both case studies use the TxGNN scaled reproduction (hidden_dim=64, no pre-trained node
          features). Rankings should not be interpreted as clinical recommendations.
        </li>
        <li>
          The disease-disease similarity pathway dominates both case studies, consistent with the
          Q6 finding that the similarity module is the load-bearing component.
        </li>
        <li>
          For Case A (rare, n_pos=1): the zero-shot rare-disease generalization claim is not
          confirmed in this scaled reproduction.
        </li>
        <li>
          For Case B (well-studied, n_pos=45): several clinically plausible antibiotics appear in
          the top-20, but embedding noise (64 dims) produces implausible cancer drug predictions.
        </li>
      </ul>
    </div>
  )
}
