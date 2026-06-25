# DESIGN.md — Visual Redesign Spec

## Token System

| Token | Value | Use |
|-------|-------|-----|
| `--paper` | `#F5F8FB` | Page background |
| `--ink` | `#14202E` | Primary text |
| `--structure` | `#5C6E80` | Edges, rules, secondary text |
| `--accent` | `#0F766E` | Clinical teal — links, focus, KG path |
| `--indication` | `#1F8A5B` | Semantic green — indication data ONLY |
| `--contra` | `#B5573A` | Semantic clay — contraindication data ONLY |
| `--paper-raised` | `#FFFFFF` | White card surface |
| `--border` | `rgba(92,110,128,0.2)` | Subtle rule |
| `--border-strong` | `rgba(92,110,128,0.4)` | Stronger rule |
| `--accent-muted` | `rgba(15,118,110,0.08)` | Teal hover/focus tint |
| `--indication-muted` | `rgba(31,138,91,0.1)` | Indication header tint |
| `--contra-muted` | `rgba(181,87,58,0.1)` | Contra header tint |
| `--shadow-sm` | (see CSS) | Subtle card lift |
| `--shadow-md` | (see CSS) | Moderate card lift |
| `--radius` | `6px` | Default border radius |
| `--radius-lg` | `10px` | Larger radius |
| `--mono-nums` | IBM Plex Mono | Numeric display font |

## Typography

- **Display:** Space Grotesk 500/700 — headings, brand, numbers
- **Body:** Inter 400/500/600 — prose, nav labels
- **Mono:** IBM Plex Mono 400/500 — all metrics, provenance, file paths

All metric values rendered in `var(--font-mono)` with `font-variant-numeric: tabular-nums`.

## Layout

### Left Index Rail (desktop ≥768px)
```
┌─────────────────────────────────────────────────────────────┐
│ [Rail 200px]  │  [Main content, max-width 860px]            │
│               │                                             │
│  TxGNN        │  h1 Page Title                              │
│               │                                             │
│  01 Overview  │  Content area                               │
│  02 Methods   │                                             │
│  03 Q1·KG     │                                             │
│  ...          │                                             │
│               │                                             │
└─────────────────────────────────────────────────────────────┘
```

### Mobile (≤768px)
```
┌─────────────────────────────────────────────────────────────┐
│ [Horizontal rail strip, 44px sticky, scroll-x]             │
│  01 Overview  02 Methods  03 Q1·KG  ...                     │
├─────────────────────────────────────────────────────────────┤
│  Main content (full width)                                  │
└─────────────────────────────────────────────────────────────┘
```

## PathRenderer Component

SVG component for multi-hop KG paths.
- Nodes: labeled circles (disease=contra clay, drug=indication green, other=accent teal)
- Edges: horizontal arrows, labeled in mono 0.65rem
- Animation: stroke-dasharray reveal, 800ms (gated behind prefers-reduced-motion)
- Layout: fixed horizontal, nodes evenly spaced
- Width: 600px for 3-4 nodes, wider for 5+

## Finding Boxes

```
.finding-box         — teal left border + muted teal bg
.finding-box.indication — green left border + muted green bg  
.finding-box.contra     — clay left border + muted clay bg
```

## Section Divider Motif

Small SVG: two circles connected by teal stroke (~48×16px). Replaces `<hr>`.

## ASCII Wireframes

### Hero (Overview)

```
TxGNN Drug Repurposing
A scaled reproduction on PrimeKG · RTX 4060 8GB

[○]──gene_associated──[○]──treats──[○]
 FHC                 sarcomere     Propranolol (predicted)

source: results/predictions/case_study_caseA_paths_txgnn.csv

01  Does KG augmentation improve over no-KG?
    KG message passing does not improve indication AUPRC...

02  Is there a better alternative to two-phase training?
    No. Two-phase TxGNN leads on zero-shot...
```

### Results Table

```
                    ┌─ Indication (green header) ─┐  ┌─ Contra (clay header) ─┐
Model       Split   AUPRC(scaled)  AUPRC(paper)    AUPRC(scaled)  AUPRC(paper)
─────────────────────────────────────────────────────────────────────────────
GNN no-KG   std     0.893 ± 0.006  —               0.945 ± 0.002  —
TxGNN 2p    zero    0.736 ± 0.011  — [no data]      0.832 ± 0.010  — [no data]
```

### Case Study Path

```
[FHC]────gene_assoc────[MYH7]────drug_target────[Propranolol]
  disease                gene                      drug
```
