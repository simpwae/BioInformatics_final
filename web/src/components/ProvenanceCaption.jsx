import React from 'react'

/**
 * Renders a "source: path/to/file" caption below a chart or table.
 * Every figure and table must use this component.
 */
export default function ProvenanceCaption({ source, extra }) {
  if (!source) return null
  return (
    <p className="provenance" aria-label={`Source: ${source}`}>
      {source}
      {extra && <span style={{ fontFamily: 'inherit', fontWeight: 'normal' }}> — {extra}</span>}
    </p>
  )
}
