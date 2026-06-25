import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Overview', exact: true },
  { to: '/methods', label: 'Methods' },
  { to: '/results', label: 'Q1 + Q5 Results' },
  { to: '/q2-alternatives', label: 'Q2 Alternatives' },
  { to: '/q3-zeroshot', label: 'Q3 Zero-Shot' },
  { to: '/q6-ablations', label: 'Q6 Ablations' },
  { to: '/case-studies', label: 'Q4 Case Studies' },
  { to: '/report', label: 'Full Report' },
  { to: '/evidence', label: 'Evidence' },
]

export default function Nav() {
  const location = useLocation()

  return (
    <nav aria-label="Main navigation" className="site-nav">
      <div className="site-nav-inner">
        <NavLink to="/" className="site-nav-brand" aria-label="Home — TxGNN Case Study">
          TxGNN Case Study
        </NavLink>
        <ul className="site-nav-links" role="list">
          {NAV_ITEMS.map(({ to, label, exact }) => {
            const isActive = exact ? location.pathname === to : location.pathname.startsWith(to)
            return (
              <li key={to}>
                <NavLink
                  to={to}
                  end={exact}
                  className="site-nav-link"
                  aria-current={isActive ? 'page' : undefined}
                >
                  {label}
                </NavLink>
              </li>
            )
          })}
        </ul>
      </div>
    </nav>
  )
}
