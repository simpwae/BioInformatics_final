import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'

const RAIL_ITEMS = [
  { to: '/',               num: '01', label: 'Overview',    exact: true },
  { to: '/methods',        num: '02', label: 'Methods' },
  { to: '/results',        num: '03', label: 'Q1 · KG vs No-KG' },
  { to: '/q2-alternatives',num: '04', label: 'Q2 · Training' },
  { to: '/q3-zeroshot',   num: '05', label: 'Q3 · Zero-shot' },
  { to: '/case-studies',  num: '06', label: 'Q4 · Cases' },
  { to: '/results',        num: '07', label: 'Q5 · Table',   skipDupe: true },
  { to: '/report',         num: '08', label: 'Report' },
  { to: '/evidence',       num: '09', label: 'Evidence' },
]

export default function Rail() {
  const location = useLocation()

  return (
    <nav aria-label="Site navigation" className="site-rail">
      <NavLink to="/" className="site-rail-brand" aria-label="Home — TxGNN">
        TxGNN
      </NavLink>
      <ul className="site-rail-links" role="list">
        {RAIL_ITEMS.map(({ to, num, label, exact, skipDupe }) => {
          // Q5 shares the /results route with Q1 — mark it active only when already on results
          const isActive = exact
            ? location.pathname === to
            : location.pathname === to || location.pathname.startsWith(to + '/')

          return (
            <li key={`${num}-${to}`} className="site-rail-item">
              <NavLink
                to={to}
                end={exact}
                className="site-rail-link"
                aria-current={isActive ? 'page' : undefined}
              >
                <span className="rail-num">{num}</span>
                <span className="rail-label">{label}</span>
              </NavLink>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
