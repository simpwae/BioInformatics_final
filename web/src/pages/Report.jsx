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
    border: '1px solid #d0d0d0',
    borderBottom: active ? '1px solid #fff' : '1px solid #d0d0d0',
    borderRadius: '4px 4px 0 0',
    cursor: 'pointer',
    background: active ? '#fff' : '#f4f4f4',
    fontSize: '0.875rem',
    fontWeight: active ? 600 : 400,
    marginRight: '0.25rem',
    color: active ? '#1a4e8a' : '#333',
  })

  return (
    <div className="page">
      <h1>Full Report</h1>
      <p>
        The five report sections, rendered from <code>report/sections/</code>. Content is loaded
        from files — not typed into this page.
      </p>

      <div style={{ display: 'flex', flexWrap: 'wrap', borderBottom: '1px solid #d0d0d0', marginBottom: '0' }}>
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
            border: '1px solid #d0d0d0',
            borderTop: 'none',
            padding: '1.5rem',
            borderRadius: '0 0 4px 4px',
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
        .report-content h1 { font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }
        .report-content h2 { font-size: 1.2rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }
        .report-content h3 { font-size: 1rem; margin-top: 1rem; margin-bottom: 0.4rem; }
        .report-content p { margin-bottom: 0.75rem; }
        .report-content ul, .report-content ol { padding-left: 1.5rem; margin-bottom: 0.75rem; }
        .report-content li { margin-bottom: 0.25rem; }
        .report-content table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-bottom: 1rem; }
        .report-content th { background: #f2f4f7; font-weight: 600; padding: 0.4rem 0.6rem; border: 1px solid #d0d0d0; }
        .report-content td { padding: 0.35rem 0.6rem; border: 1px solid #d0d0d0; }
        .report-content blockquote { border-left: 3px solid #1a4e8a; background: #e8f0fb; padding: 0.75rem 1rem; margin: 0 0 0.75rem 0; border-radius: 0 4px 4px 0; }
        .report-content blockquote p { margin-bottom: 0; }
        .report-content code { font-family: monospace; font-size: 0.875em; background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 2px; }
        .report-content pre { background: #f4f4f4; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; margin-bottom: 0.75rem; }
        .report-content pre code { background: none; padding: 0; }
        .report-content strong { font-weight: 600; }
        .report-content em { font-style: italic; }
        .report-content hr { border: none; border-top: 1px solid #d0d0d0; margin: 1.5rem 0; }
      `}</style>
    </div>
  )
}
