import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'

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

const navStyle = {
  borderBottom: '1px solid #d0d0d0',
  background: '#fff',
  position: 'sticky',
  top: 0,
  zIndex: 50,
}

const innerStyle = {
  maxWidth: '1100px',
  margin: '0 auto',
  padding: '0 1.5rem',
  display: 'flex',
  alignItems: 'center',
  gap: '0',
  flexWrap: 'wrap',
}

const brandStyle = {
  fontWeight: 700,
  fontSize: '0.95rem',
  color: '#111',
  textDecoration: 'none',
  padding: '0.75rem 0',
  marginRight: '1.5rem',
  whiteSpace: 'nowrap',
}

const linkStyle = {
  padding: '0.75rem 0.6rem',
  fontSize: '0.85rem',
  color: '#333',
  textDecoration: 'none',
  borderBottom: '3px solid transparent',
  whiteSpace: 'nowrap',
}

const activeLinkStyle = {
  ...linkStyle,
  color: '#1a4e8a',
  borderBottom: '3px solid #1a4e8a',
  fontWeight: 600,
}

export default function Nav() {
  return (
    <nav aria-label="Main navigation" style={navStyle}>
      <div style={innerStyle}>
        <NavLink to="/" style={brandStyle} aria-label="Home — Drug Repurposing KG">
          Drug Repurposing KG
        </NavLink>
        {NAV_ITEMS.map(({ to, label, exact }) => (
          <NavLink
            key={to}
            to={to}
            end={exact}
            style={({ isActive }) => (isActive ? activeLinkStyle : linkStyle)}
            aria-current={undefined}
          >
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
