import React from 'react'
import { useData } from '../context/DataContext.jsx'

export default function Footer() {
  const { data } = useData()

  return (
    <footer className="site-footer" role="contentinfo">
      <div className="site-footer-inner">
        <span>Generated from <code>results/</code> &mdash; data is build-time, not hardcoded</span>
        {data?.generatedAt && (
          <span>
            Generated: <time dateTime={data.generatedAt}>{data.generatedAt}</time>
          </span>
        )}
      </div>
    </footer>
  )
}
