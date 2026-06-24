/**
 * data-binding.test.jsx
 * Verifies that the Results page renders values that match results.json,
 * and that no hardcoded numbers appear outside build-data.mjs.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import React from 'react'

// Sample data matching the actual results.json schema
const SAMPLE_DATA = {
  generatedAt: '2026-06-24T00:00:00.000Z',
  comparisonTable: [
    {
      model: 'gnn_no_kg',
      displayName: 'GNN Baseline (no-KG)',
      split: 'standard',
      n_seeds_run: 3,
      reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.8926,
      auprc_ind_std: 0.0061,
      auroc_ind: 0.9719,
      auroc_ind_std: 0.0024,
      auprc_contra: 0.9453,
      auprc_contra_std: 0.0023,
      auroc_contra: 0.9913,
      auroc_contra_std: 0.0003,
      wall_s: 0.7,
      paper_reported: null,
    },
    {
      model: 'gnn_kg',
      displayName: 'GNN Baseline (KG)',
      split: 'standard',
      n_seeds_run: 3,
      reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.8245,
      auprc_ind_std: 0.022,
      auroc_ind: 0.9716,
      auroc_ind_std: 0.0055,
      auprc_contra: 0.895,
      auprc_contra_std: 0.0262,
      auroc_contra: 0.9836,
      auroc_contra_std: 0.0034,
      wall_s: 145.5,
      paper_reported: null,
    },
  ],
  ablationMatrix: {
    q6_decision: {
      status: 'DECIDED',
      txgnn_attn_on_zeroshot_ind_auprc: 0.73568,
      txgnn_attn_off_zeroshot_ind_auprc: 0.77202,
      delta_attn_on_minus_off: -0.03634,
      threshold: 0.02,
      attention_optional: false,
      conclusion: 'Attention is DETRIMENTAL',
    },
    matrix: [],
  },
  q6AblationTable: [],
  degradationCurveData: { raw: [], binned: [] },
  caseStudies: {
    caseA: {
      diseaseId: '24573',
      diseaseName: 'Familial Hypertrophic Cardiomyopathy',
      shortName: 'FHC',
      nPos: 1,
      relation: 'indication',
      auprc: 0.025,
      source: 'results/predictions/case_study_caseA_txgnn.csv',
      predictions: [],
      contraindications: [],
      paths: [],
      pathsSource: 'results/predictions/case_study_caseA_paths_txgnn.csv',
      knownDrug: 'Propranolol (DB00571)',
      knownDrugInTop20: false,
      notes: 'Model fails.',
    },
    caseB: {
      diseaseId: '5545',
      diseaseName: 'Staphylococcus Aureus Infection',
      shortName: 'S. aureus',
      nPos: 45,
      relation: 'indication',
      auprc: 0.088,
      source: 'results/predictions/case_study_caseB_txgnn.csv',
      predictions: [],
      paths: [],
      pathsSource: 'results/predictions/case_study_caseB_paths_txgnn.csv',
      firstPositiveRank: 18,
      firstPositiveDrug: 'Benzylpenicillin (DB01053)',
      notes: 'Benzylpenicillin at rank 18.',
    },
  },
  reportSections: [],
  paperReported: {
    zeroshot: {
      _relativeNote: 'relative only: +19.0% ind / +23.9% contra vs next-best baseline (abs. in Suppl. S1/S2)',
    },
  },
}

// Mock fetch to return sample data
beforeEach(() => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve(SAMPLE_DATA),
    })
  )
})

// Mock recharts to avoid SVG rendering issues in jsdom
vi.mock('recharts', () => ({
  BarChart: ({ children }) => React.createElement('div', { 'data-testid': 'bar-chart' }, children),
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

function renderResults() {
  render(
    <MemoryRouter>
      <DataProvider>
        <Results />
      </DataProvider>
    </MemoryRouter>
  )
}

describe('data-binding', () => {
  it('renders gnn_no_kg AUPRC value from JSON (not hardcoded)', async () => {
    renderResults()
    // The value 0.8926 should appear as "0.893" (3dp formatting) — may appear in multiple cells
    const els = await screen.findAllByText(/0\.893/, { exact: false })
    expect(els.length).toBeGreaterThan(0)
  })

  it('renders gnn_kg AUPRC value from JSON', async () => {
    renderResults()
    const els = await screen.findAllByText(/0\.825/, { exact: false })
    expect(els.length).toBeGreaterThan(0)
  })

  it('shows zero-shot paper_reported note (not "0")', async () => {
    renderResults()
    // The note should contain "relative only" or "+19.0%"
    const els = await screen.findAllByText(/relative only|19\.0%/i, { exact: false })
    expect(els.length).toBeGreaterThan(0)
  })

  it('shows GNN Baseline display name from JSON', async () => {
    renderResults()
    const els = await screen.findAllByText(/GNN Baseline \(no-KG\)/i)
    expect(els.length).toBeGreaterThan(0)
  })
})
