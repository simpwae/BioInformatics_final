import React from 'react'
import { useData } from '../context/DataContext.jsx'

export default function Footer() {
  const { data } = useData()

  return (
    <footer
      style={{
        borderTop: '1px solid #d0d0d0',
        marginTop: '3rem',
        padding: '1.5rem',
        fontSize: '0.8rem',
        color: '#555',
        textAlign: 'center',
      }}
      role="contentinfo"
    >
      <p style={{ marginBottom: '0.25rem' }}>
        Scaled reproduction of TxGNN (Huang et al., 2024) on PrimeKG. RTX 4060, 8 GB VRAM.
        Seeds [42, 0, 1]. All results in <code>results/</code>.
      </p>
      {data?.generatedAt && (
        <p style={{ marginBottom: 0 }}>
          Data generated: <time dateTime={data.generatedAt}>{data.generatedAt}</time>
        </p>
      )}
    </footer>
  )
}
