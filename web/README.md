# Drug Repurposing Web Interface

Static React web app for the TxGNN drug repurposing bioinformatics project.
Scaled reproduction of Huang et al. (2024), Nature Medicine.

## Setup

From the `web/` directory:

```bash
npm install
npm run build:data   # generate public/data/results.json from ../results/
npm run dev          # development server
npm run build        # production build (runs build:data first via prebuild)
npm run test         # run tests
npm run preview      # preview production build
```

## Architecture

- **Vite + React 18** — static site, no server-side code
- **React Router v6** — client-side routing
- **Recharts** — charts (all grayscale-compatible)
- **react-markdown** — renders report sections from .md files
- **Context API** — single fetch of `/data/results.json` on mount, shared via DataProvider

## Data Flow

1. `scripts/build-data.mjs` reads all result files from `../results/` and report sections from `../report/sections/`
2. Outputs `public/data/results.json`
3. The React app fetches this file once and provides it via DataContext

**The only place `paper_reported` values are hardcoded is `scripts/build-data.mjs`.**
No .jsx file may contain hardcoded metric values.

## Routes

| Route | Page | Content |
|-------|------|---------|
| `/` | Overview | Project summary, 6 research questions |
| `/methods` | Methods | Data, models, evaluation protocol |
| `/results` | Results | Q1 (KG vs no-KG) + Q5 (full comparison table) |
| `/q2-alternatives` | Q2 Alternatives | Two-phase vs single-stage vs joint contrastive |
| `/q3-zeroshot` | Q3 Zero-Shot | Why zero-shot, degradation curve |
| `/q6-ablations` | Q6 Ablations | Attention ablation (DETRIMENTAL finding) |
| `/case-studies` | Case Studies | Q4: FHC (rare) and S. aureus (well-studied) |
| `/report` | Full Report | All 5 report sections as tabbed markdown |
| `/evidence` | Evidence | Citations, sourcing, claims register |

## Provenance

Every chart and table has a `source: path/to/file` caption (ProvenanceCaption component).
No numbers are typed manually anywhere in the .jsx files.

## Deployment (Vercel)

```
Build command: npm run build:data && npm run build
Output directory: dist
Framework: vite
```

The `web/` directory should be set as the root directory in Vercel settings.

## Hardware Note

All models trained on NVIDIA GeForce RTX 4060, 8 GB VRAM.
`hidden_dim=64` (published: 512) due to VRAM constraint.
All results labeled `scaled_reproduction`.
