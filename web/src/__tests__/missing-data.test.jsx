/**
 * missing-data.test.jsx
 * Verifies that zero-shot paper_reported renders explicit text, never "0".
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import React from 'react'

// Data with zero-shot rows where paper_reported has the relative-only note
const ZEROSHOT_DATA = {
  generatedAt: '2026-06-24T00:00:00.000Z',
  paperReported: {
    zeroshot: {
      _relativeNote:
        'relative only: +19.0% ind / +23.9% contra vs next-best baseline (abs. in Suppl. S1/S2)',
      txgnn_two_phase: {
        auprc_ind: '0.90 ± 0.02',
        auprc_contra: '0.80 ± 0.01',
        source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2',
      },
    },
    standard: {},
  },
  comparisonTable: [
    {
      model: 'txgnn_two_phase',
      displayName: 'TxGNN Two-Phase',
      split: 'zeroshot',
      n_seeds_run: 3,
      reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.7357,
      auprc_ind_std: 0.0108,
      auroc_ind: 0.9503,
      auroc_ind_std: 0.0094,
      auprc_contra: 0.8316,
      auprc_contra_std: 0.0102,
      auroc_contra: 0.9734,
      auroc_contra_std: 0.0023,
      wall_s: 71.6,
      paper_reported: {
        auprc_ind: '0.90 ± 0.02',
        auprc_contra: '0.80 ± 0.01',
        source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2',
      },
    },
    {
      model: 'gnn_no_kg',
      displayName: 'GNN Baseline (no-KG)',
      split: 'zeroshot',
      n_seeds_run: 3,
      reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.7044,
      auprc_ind_std: 0.0272,
      auroc_ind: 0.9342,
      auroc_ind_std: 0.0054,
      auprc_contra: 0.7252,
      auprc_contra_std: 0.0079,
      auroc_contra: 0.9315,
      auroc_contra_std: 0.0057,
      wall_s: 0.2,
      paper_reported: null,
    },
  ],
  ablationMatrix: { q6_decision: {}, matrix: [] },
  q6AblationTable: [],
  degradationCurveData: { raw: [], binned: [] },
  caseStudies: {
    caseA: {
      diseaseId: '24573',
      diseaseName: 'Familial Hypertrophic Cardiomyopathy',
      nPos: 1,
      auprc: 0.025,
      source: 'results/predictions/case_study_caseA_txgnn.csv',
      predictions: [],
      contraindications: [],
      paths: [],
      pathsSource: 'results/predictions/case_study_caseA_paths_txgnn.csv',
      knownDrug: 'Propranolol',
      knownDrugInTop20: false,
      notes: '',
    },
    caseB: {
      diseaseId: '5545',
      diseaseName: 'Staphylococcus Aureus Infection',
      nPos: 45,
      auprc: 0.088,
      source: 'results/predictions/case_study_caseB_txgnn.csv',
      predictions: [],
      paths: [],
      pathsSource: 'results/predictions/case_study_caseB_paths_txgnn.csv',
      firstPositiveRank: 18,
      firstPositiveDrug: 'Benzylpenicillin (DB01053)',
      notes: '',
    },
  },
  reportSections: [],
}

beforeEach(() => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve(ZEROSHOT_DATA),
    })
  )
})

vi.mock('recharts', () => ({
  BarChart: ({ children }) => React.createElement('div', {}, children),
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }) => React.createElement('div', {}, children),
  ReferenceLine: () => null,
  ScatterChart: ({ children }) => React.createElement('div', {}, children),
  Scatter: () => null,
  ErrorBar: () => null,
}))

import { DataProvider } from '../context/DataContext.jsx'
import Results from '../pages/Results.jsx'

describe('missing-data / paper_reported rendering', () => {
  it('renders the relative-only note for zero-shot paper_reported, not "0"', async () => {
    const { findByText } = render(
      <MemoryRouter>
        <DataProvider>
          <Results />
        </DataProvider>
      </MemoryRouter>
    )

    // Should find the "relative only" note
    const el = await findByText(/relative only/i, { exact: false })
    expect(el).toBeTruthy()

    // Should NOT find bare "0" as a paper_reported value in the zero-shot paper cell
    // (the actual zero-shot AUPRC "0.736" may appear, but the paper_reported
    // cell must not contain just "0")
    const container = el.closest('body')
    const cells = container ? container.querySelectorAll('td') : []
    const hasRawZero = Array.from(cells).some((td) => td.textContent.trim() === '0')
    expect(hasRawZero).toBe(false)
  })

  it('renders TxGNN paper_reported AUPRC (0.90) for zero-shot', async () => {
    const { findByText } = render(
      <MemoryRouter>
        <DataProvider>
          <Results />
        </DataProvider>
      </MemoryRouter>
    )
    // The paper_reported value 0.90 ± 0.02 should appear somewhere
    const el = await findByText(/0\.90/, { exact: false })
    expect(el).toBeTruthy()
  })

  it('does not show [NOT YET RUN] rows in the table cells', async () => {
    const { findByText, container } = render(
      <MemoryRouter>
        <DataProvider>
          <Results />
        </DataProvider>
      </MemoryRouter>
    )
    // Wait for data to load
    await findByText(/TxGNN Two-Phase/i)
    // Should not have NOT YET RUN in any table data cell
    const cells = container.querySelectorAll('td')
    const hasNYR = Array.from(cells).some((td) => td.textContent.includes('[NOT YET RUN]'))
    expect(hasNYR).toBe(false)
  })
})
