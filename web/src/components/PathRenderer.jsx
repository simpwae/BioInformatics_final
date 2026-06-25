import React, { useId } from 'react'

/**
 * PathRenderer — draws a multi-hop KG path as an SVG.
 *
 * Props:
 *   nodes   [{id, label, type: 'disease'|'gene'|'protein'|'drug'|'other'}]
 *   edges   [{source, target, label}]
 *   animate boolean (default true, gated behind prefers-reduced-motion)
 */

const NODE_COLORS = {
  disease: '#B5573A',  // --contra clay
  drug:    '#1F8A5B',  // --indication green
  gene:    '#0F766E',  // --accent teal
  protein: '#0F766E',
  other:   '#5C6E80',  // --structure
}

const NODE_RADIUS = 22
const H_PAD       = 60
const V_CENTER     = 50
const EDGE_LABEL_Y = 30
const MIN_NODE_GAP = 140

function nodeX(index, total, svgWidth) {
  const usable = svgWidth - H_PAD * 2
  if (total === 1) return svgWidth / 2
  return H_PAD + (index / (total - 1)) * usable
}

export default function PathRenderer({ nodes = [], edges = [], animate = true }) {
  const uid = useId()
  const prefersReduced =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches

  const shouldAnimate = animate && !prefersReduced

  const n = nodes.length
  if (n === 0) return null

  const svgWidth  = Math.max(600, n * MIN_NODE_GAP + H_PAD * 2)
  const svgHeight = 100

  // Compute node positions
  const positions = nodes.map((node, i) => ({
    x: nodeX(i, n, svgWidth),
    y: V_CENTER,
  }))

  // Total path length for animation (rough estimate)
  const totalPathLength = edges.length > 0 ? (positions[n - 1].x - positions[0].x) : 0

  return (
    <div
      style={{ overflowX: 'auto', marginBottom: '0.75rem' }}
      role="img"
      aria-label={`KG path: ${nodes.map((nd) => nd.label).join(' → ')}`}
    >
      <svg
        width={svgWidth}
        height={svgHeight}
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        xmlns="http://www.w3.org/2000/svg"
        style={{ display: 'block', minWidth: svgWidth }}
      >
        <defs>
          {/* Arrowhead marker */}
          <marker
            id={`${uid}-arrow`}
            markerWidth="8"
            markerHeight="8"
            refX="7"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L8,3 z" fill="var(--accent, #0F766E)" />
          </marker>

          {/* Clip path for animation reveal */}
          {shouldAnimate && (
            <clipPath id={`${uid}-reveal`}>
              <rect x="0" y="0" width="0" height={svgHeight}>
                <animate
                  attributeName="width"
                  from="0"
                  to={svgWidth}
                  dur="0.8s"
                  begin="0.1s"
                  fill="freeze"
                  calcMode="ease-in-out"
                />
              </rect>
            </clipPath>
          )}
        </defs>

        {/* Edges */}
        <g clipPath={shouldAnimate ? `url(#${uid}-reveal)` : undefined}>
          {edges.map((edge, i) => {
            const srcIdx = nodes.findIndex((nd) => nd.id === edge.source)
            const tgtIdx = nodes.findIndex((nd) => nd.id === edge.target)
            if (srcIdx === -1 || tgtIdx === -1) return null

            const x1 = positions[srcIdx].x + NODE_RADIUS
            const x2 = positions[tgtIdx].x - NODE_RADIUS - 6 /* arrowhead space */
            const y1 = positions[srcIdx].y
            const y2 = positions[tgtIdx].y
            const midX = (x1 + x2) / 2

            return (
              <g key={i}>
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="var(--accent, #0F766E)"
                  strokeWidth="1.5"
                  markerEnd={`url(#${uid}-arrow)`}
                />
                {edge.label && (
                  <text
                    x={midX}
                    y={EDGE_LABEL_Y}
                    textAnchor="middle"
                    fontSize="9"
                    fontFamily="var(--font-mono, monospace)"
                    fill="var(--structure, #5C6E80)"
                  >
                    {edge.label}
                  </text>
                )}
              </g>
            )
          })}
        </g>

        {/* Nodes (drawn on top, not clipped so they appear immediately) */}
        {nodes.map((node, i) => {
          const { x, y } = positions[i]
          const color = NODE_COLORS[node.type] || NODE_COLORS.other
          return (
            <g key={node.id || i}>
              <circle
                cx={x}
                cy={y}
                r={NODE_RADIUS}
                fill={color}
                fillOpacity="0.15"
                stroke={color}
                strokeWidth="1.5"
              />
              <text
                x={x}
                y={y + 4}
                textAnchor="middle"
                fontSize="9"
                fontFamily="var(--font-mono, monospace)"
                fontWeight="500"
                fill={color}
              >
                {node.label.length > 12 ? node.label.slice(0, 11) + '…' : node.label}
              </text>
              {/* Type label below node */}
              <text
                x={x}
                y={y + NODE_RADIUS + 14}
                textAnchor="middle"
                fontSize="8"
                fontFamily="var(--font-mono, monospace)"
                fill="var(--structure, #5C6E80)"
              >
                {node.type}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
