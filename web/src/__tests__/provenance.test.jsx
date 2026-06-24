/**
 * provenance.test.jsx
 * Every chart/table component must render a non-empty "source:" caption.
 */
import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import React from 'react'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

describe('ProvenanceCaption', () => {
  it('renders the source path', () => {
    const { container } = render(
      <ProvenanceCaption source="results/metrics/comparison_table.csv" />
    )
    const el = container.querySelector('.provenance')
    expect(el).not.toBeNull()
    expect(el.textContent).toContain('results/metrics/comparison_table.csv')
  })

  it('includes the "source:" prefix (via ::before or aria-label)', () => {
    const { container } = render(
      <ProvenanceCaption source="results/ablations/matrix.json" />
    )
    const el = container.querySelector('.provenance')
    expect(el).not.toBeNull()
    // aria-label contains "Source:" (note capital S from aria-label)
    const ariaLabel = el.getAttribute('aria-label')
    expect(ariaLabel).toContain('Source:')
    expect(ariaLabel).toContain('results/ablations/matrix.json')
  })

  it('renders extra text when provided', () => {
    const { container } = render(
      <ProvenanceCaption
        source="results/metrics/comparison_table.csv"
        extra="zero-shot split, seeds [42, 0, 1]"
      />
    )
    const el = container.querySelector('.provenance')
    expect(el.textContent).toContain('zero-shot split')
  })

  it('renders nothing when source is empty', () => {
    const { container } = render(<ProvenanceCaption source="" />)
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when source is undefined', () => {
    const { container } = render(<ProvenanceCaption />)
    expect(container.firstChild).toBeNull()
  })

  it('source caption is non-empty for a real path', () => {
    const paths = [
      'results/metrics/comparison_table.csv',
      'results/ablations/matrix.json',
      'results/metrics/degradation_curve_data.json',
      'results/predictions/case_study_caseA_txgnn.csv',
      'results/predictions/case_study_caseB_txgnn.csv',
    ]
    paths.forEach((path) => {
      const { container } = render(<ProvenanceCaption source={path} />)
      const el = container.querySelector('.provenance')
      expect(el).not.toBeNull()
      expect(el.textContent.trim().length).toBeGreaterThan(0)
    })
  })
})
