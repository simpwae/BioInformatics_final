/**
 * markdown.test.jsx
 * Verifies that the Report page renders all 5 sections with content.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import React from 'react'

const SAMPLE_SECTIONS = [
  {
    filename: '01_introduction.md',
    title: 'Introduction',
    content: '# 1. Introduction\n\nDrug repurposing — finding new therapeutic uses for approved drugs.',
    source: 'report/sections/01_introduction.md',
  },
  {
    filename: '02_methods.md',
    title: 'Methods',
    content: '# 2. Methods\n\nPrimeKG downloaded from Harvard Dataverse.',
    source: 'report/sections/02_methods.md',
  },
  {
    filename: '03_results.md',
    title: 'Results',
    content: '# 3. Results\n\nAll numbers link to files in `results/`.',
    source: 'report/sections/03_results.md',
  },
  {
    filename: '04_case_studies.md',
    title: 'Case Studies',
    content: '# 4. Case Studies (Q4)\n\nSelected before examining predictions.',
    source: 'report/sections/04_case_studies.md',
  },
  {
    filename: '05_discussion.md',
    title: 'Discussion',
    content: '# 5. Discussion\n\nKG augmentation does not consistently help.',
    source: 'report/sections/05_discussion.md',
  },
]

const SAMPLE_DATA = {
  generatedAt: '2026-06-24T00:00:00.000Z',
  paperReported: { zeroshot: { _relativeNote: 'relative only' }, standard: {} },
  comparisonTable: [],
  ablationMatrix: { q6_decision: {}, matrix: [] },
  q6AblationTable: [],
  degradationCurveData: { raw: [], binned: [] },
  caseStudies: {
    caseA: { diseaseId: '24573', diseaseName: 'FHC', nPos: 1, auprc: 0.025, source: 's', predictions: [], contraindications: [], paths: [], pathsSource: 's', knownDrug: 'P', knownDrugInTop20: false, notes: '' },
    caseB: { diseaseId: '5545', diseaseName: 'S. aureus', nPos: 45, auprc: 0.088, source: 's', predictions: [], paths: [], pathsSource: 's', firstPositiveRank: 18, firstPositiveDrug: 'B', notes: '' },
  },
  reportSections: SAMPLE_SECTIONS,
}

beforeEach(() => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve(SAMPLE_DATA),
    })
  )
})

import { DataProvider } from '../context/DataContext.jsx'
import Report from '../pages/Report.jsx'

describe('markdown / report sections', () => {
  it('renders all 5 section tabs', async () => {
    const { findByText } = render(
      <MemoryRouter>
        <DataProvider>
          <Report />
        </DataProvider>
      </MemoryRouter>
    )

    // All 5 tab labels should be visible
    for (const section of SAMPLE_SECTIONS) {
      const el = await findByText(section.title)
      expect(el).toBeTruthy()
    }
  })

  it('renders the default (first) section content as markdown', async () => {
    const { findByText } = render(
      <MemoryRouter>
        <DataProvider>
          <Report />
        </DataProvider>
      </MemoryRouter>
    )

    // The first section's text content should be rendered
    const el = await findByText(/Drug repurposing/i, { exact: false })
    expect(el).toBeTruthy()
  })

  it('renders source provenance caption for the first section', async () => {
    const { findAllByClass, container, findByText } = render(
      <MemoryRouter>
        <DataProvider>
          <Report />
        </DataProvider>
      </MemoryRouter>
    )

    // Wait for load
    await findByText('Introduction')

    // Check that a provenance caption exists
    const provenanceEls = container.querySelectorAll('.provenance')
    expect(provenanceEls.length).toBeGreaterThan(0)
    const firstCaption = provenanceEls[0]
    expect(firstCaption.textContent).toContain('report/sections/')
  })

  it('shows exactly 5 section tabs (not fewer, not more)', async () => {
    const { findByText, container } = render(
      <MemoryRouter>
        <DataProvider>
          <Report />
        </DataProvider>
      </MemoryRouter>
    )

    // Wait for data
    await findByText('Introduction')

    // Count buttons in the tab bar (should be exactly 5)
    const buttons = container.querySelectorAll('button')
    // The tab buttons correspond to the 5 sections
    expect(buttons.length).toBe(5)
  })
})
