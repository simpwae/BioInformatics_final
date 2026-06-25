import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useData } from '../context/DataContext.jsx'
import ProvenanceCaption from '../components/ProvenanceCaption.jsx'

export default function Report() {
  const { data, loading, error } = useData()
  const [activeSection, setActiveSection] = useState(0)

  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const sections = data.reportSections || []

  const tabStyle = (active) => ({
    padding: '0.4rem 1rem',
    border: '1px solid var(--border)',
    borderBottom: active ? '1px solid var(--paper-raised)' : '1px solid var(--border)',
    borderRadius: 'var(--radius) var(--radius) 0 0',
    cursor: 'pointer',
    background: active ? 'var(--paper-raised)' : 'var(--paper)',
    fontSize: '0.8125rem',
    fontFamily: 'var(--font-body)',
    fontWeight: active ? 600 : 400,
    marginRight: '0.25rem',
    color: active ? 'var(--accent)' : 'var(--structure)',
  })

  return (
    <div className="page">
      <h1>Full Report</h1>
      <p>
        The five report sections, rendered from <code>report/sections/</code>. Content is loaded
        from files — not typed into this page.
      </p>

      <div style={{ display: 'flex', flexWrap: 'wrap', borderBottom: '1px solid var(--border)', marginBottom: '0' }}>
        {sections.map((s, i) => (
          <button key={i} style={tabStyle(activeSection === i)} onClick={() => setActiveSection(i)}>
            {s.title}
          </button>
        ))}
      </div>

      {sections.length === 0 && (
        <p style={{ color: '#555', marginTop: '1rem' }}>
          No report sections found. Run <code>npm run build:data</code> first.
        </p>
      )}

      {sections[activeSection] && (
        <div
          style={{
            border: '1px solid var(--border)',
            borderTop: 'none',
            padding: '1.5rem 2rem',
            borderRadius: '0 0 var(--radius) var(--radius)',
            background: 'var(--paper-raised)',
          }}
        >
          <div
            className="report-content"
            style={{
              maxWidth: '72ch',
            }}
          >
            <ReactMarkdown>{sections[activeSection].content}</ReactMarkdown>
          </div>
          <ProvenanceCaption source={sections[activeSection].source} />
        </div>
      )}

      <style>{`
        .report-content { font-family: var(--font-body); font-size: 0.9375rem; line-height: 1.65; color: var(--ink); }
        .report-content h1 { font-family: var(--font-display); font-size: 1.5rem; font-weight: 700; margin-top: 1.75rem; margin-bottom: 0.75rem; color: var(--ink); }
        .report-content h2 { font-family: var(--font-display); font-size: 1.2rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; color: var(--ink); }
        .report-content h3 { font-family: var(--font-body); font-size: 1rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.4rem; color: var(--ink); }
        .report-content p { margin-bottom: 0.75rem; max-width: 72ch; }
        .report-content ul, .report-content ol { padding-left: 1.5rem; margin-bottom: 0.75rem; }
        .report-content li { margin-bottom: 0.3rem; }
        .report-content table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.85rem; margin-bottom: 1rem; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
        .report-content th { background: var(--color-table-head); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border-strong); color: var(--structure); font-family: var(--font-body); }
        .report-content td { padding: 0.4rem 0.75rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 0.85rem; }
        .report-content tr:last-child td { border-bottom: none; }
        .report-content blockquote { border-left: 3px solid var(--accent); background: var(--accent-muted); padding: 0.75rem 1rem; margin: 0 0 0.75rem 0; border-radius: 0 var(--radius) var(--radius) 0; }
        .report-content blockquote p { margin-bottom: 0; max-width: none; }
        .report-content code { font-family: var(--font-mono); font-size: 0.875em; background: var(--accent-muted); color: var(--accent); padding: 0.1em 0.35em; border-radius: var(--radius); border: 1px solid rgba(15,118,110,0.2); }
        .report-content pre { background: var(--paper); border: 1px solid var(--border); padding: 1rem; border-radius: var(--radius); overflow-x: auto; font-size: 0.85rem; margin-bottom: 0.75rem; }
        .report-content pre code { background: none; border: none; padding: 0; color: var(--ink); }
        .report-content strong { font-weight: 600; }
        .report-content em { font-style: italic; }
        .report-content hr { border: none; border-top: 1px solid var(--border); margin: 1.75rem 0; }
      `}</style>
    </div>
  )
}
