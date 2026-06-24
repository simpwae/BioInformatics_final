/**
 * build-data.mjs
 * Reads all result files from ../results/ and ../report/sections/
 * Outputs consolidated web/public/data/results.json
 *
 * This is the ONLY place paper_reported values are hardcoded.
 * Source: Huang et al. (2024), Nature Medicine, Supplementary Tables S1 and S2 (MOESM1 ESM).
 */

import { readFileSync, writeFileSync, existsSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../../')
const resultsDir = resolve(repoRoot, 'results')
const reportDir = resolve(repoRoot, 'report/sections')
const litDir = resolve(repoRoot, 'lit')
const outputPath = resolve(__dirname, '../public/data/results.json')

// ---------------------------------------------------------------------------
// paper_reported values — ONLY sourced here; no .jsx file may hardcode these.
// Source: Huang et al. (2024), "A foundation model for clinician-centered drug
// repurposing," Nature Medicine. DOI: 10.1038/s41591-024-03233-x
// Values from Supplementary Tables S1 (indication AUPRC) and S2 (contraindication AUPRC),
// MOESM1 ESM. Retrieved and verified against both PMC11326339 (preprint) and PMC11645266.
// ---------------------------------------------------------------------------
const PAPER_REPORTED = {
  // Standard split (Suppl. Table S1 "Random" row for indication; S2 "Random" row for contraindication)
  standard: {
    txgnn_two_phase: {
      auprc_ind: '0.91 ± 0.02',
      auprc_contra: '0.82 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2',
    },
    txgnn_attn_on: {
      auprc_ind: '0.91 ± 0.02',
      auprc_contra: '0.82 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2',
    },
    han_paper: {
      auprc_ind: '0.87 ± 0.18',
      auprc_contra: '0.84 ± 0.00',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2 (HAN row)',
    },
  },
  // Zero-shot split (Suppl. Table S1/S2 "Zero-shot (random)" row)
  zeroshot: {
    txgnn_two_phase: {
      auprc_ind: '0.90 ± 0.02',
      auprc_contra: '0.80 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2',
    },
    txgnn_attn_on: {
      auprc_ind: '0.90 ± 0.02',
      auprc_contra: '0.80 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2',
    },
    // Relative-only note for zero-shot paper_reported:
    // Absolute values are in Suppl. S1/S2.
    // Main-text relative gains (Fig 2d): +19.0% ind, +23.9% contra over next-best baseline.
    _relativeNote:
      'relative only: +19.0% ind / +23.9% contra vs next-best baseline (abs. in Suppl. S1/S2)',
    bioBERT_paper: {
      auprc_ind: '0.76 ± 0.03',
      auprc_contra: null,
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1 (BioBERT row)',
    },
    rgcn_paper: {
      auprc_ind: null,
      auprc_contra: '0.64 ± 0.03',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S2 (RGCN row)',
    },
  },
}

// ---------------------------------------------------------------------------
// CSV parser (minimal, no external dependency at build time — papaparse used
// in the browser; here we parse manually)
// ---------------------------------------------------------------------------
function parseCSV(content) {
  const lines = content.trim().split('\n')
  const headers = lines[0].split(',').map((h) => h.trim())
  return lines.slice(1).map((line) => {
    // Handle quoted fields
    const values = []
    let current = ''
    let inQuotes = false
    for (const ch of line) {
      if (ch === '"') {
        inQuotes = !inQuotes
      } else if (ch === ',' && !inQuotes) {
        values.push(current.trim())
        current = ''
      } else {
        current += ch
      }
    }
    values.push(current.trim())
    const obj = {}
    headers.forEach((h, i) => {
      obj[h] = values[i] !== undefined ? values[i] : ''
    })
    return obj
  })
}

function readJSON(path) {
  if (!existsSync(path)) return null
  return JSON.parse(readFileSync(path, 'utf-8'))
}

function readCSV(path) {
  if (!existsSync(path)) return []
  return parseCSV(readFileSync(path, 'utf-8'))
}

function readText(path) {
  if (!existsSync(path)) return ''
  return readFileSync(path, 'utf-8')
}

// ---------------------------------------------------------------------------
// Model display name mapping
// ---------------------------------------------------------------------------
const MODEL_DISPLAY = {
  gnn_no_kg: 'GNN Baseline (no-KG)',
  gnn_kg: 'GNN Baseline (KG)',
  txgnn_two_phase: 'TxGNN Two-Phase',
  single_stage: 'Single-Stage',
  joint_contrastive: 'Joint Contrastive',
  txgnn_attn_on: 'TxGNN (attn=ON)',
  txgnn_attn_off: 'TxGNN (attn=OFF)',
}

// ---------------------------------------------------------------------------
// Parse comparison table — filter out [NOT YET RUN] rows
// ---------------------------------------------------------------------------
function buildComparisonTable() {
  const rows = readCSV(resolve(resultsDir, 'metrics/comparison_table.csv'))
  const valid = rows.filter(
    (r) => r.auprc_ind && r.auprc_ind !== '[NOT YET RUN]' && r.auprc_ind !== ''
  )
  return valid.map((r) => {
    const split = r.split
    const model = r.model
    const pr = PAPER_REPORTED[split]?.[model] || null

    // Parse numeric values safely
    const parseNum = (v) => {
      const n = parseFloat(v)
      return isNaN(n) ? null : n
    }

    return {
      model,
      displayName: MODEL_DISPLAY[model] || model,
      split,
      n_seeds_run: parseInt(r.n_seeds_run) || 0,
      reproduction_type: r.reproduction_type,
      auprc_ind: parseNum(r.auprc_ind),
      auprc_ind_std: parseNum(r.auprc_ind_std),
      auroc_ind: parseNum(r.auroc_ind),
      auroc_ind_std: parseNum(r.auroc_ind_std),
      auprc_contra: parseNum(r.auprc_contra),
      auprc_contra_std: parseNum(r.auprc_contra_std),
      auroc_contra: parseNum(r.auroc_contra),
      auroc_contra_std: parseNum(r.auroc_contra_std),
      wall_s: parseNum(r.wall_s),
      paper_reported: pr,
    }
  })
}

// ---------------------------------------------------------------------------
// Ablation matrix
// ---------------------------------------------------------------------------
function buildAblationMatrix() {
  return readJSON(resolve(resultsDir, 'ablations/matrix.json'))
}

// ---------------------------------------------------------------------------
// Q6 ablation table
// ---------------------------------------------------------------------------
function buildQ6AblationTable() {
  const rows = readCSV(resolve(resultsDir, 'metrics/q6_ablation_table.csv'))
  return rows.map((r) => ({
    variant: r.variant,
    displayName: MODEL_DISPLAY[r.variant] || r.variant,
    split: r.split,
    auprc_ind_mean: parseFloat(r.auprc_ind_mean),
    auprc_ind_std: parseFloat(r.auprc_ind_std),
    n_seeds: parseInt(r.n_seeds),
  }))
}

// ---------------------------------------------------------------------------
// Degradation curve data — bin by n_train_edges
// ---------------------------------------------------------------------------
function buildDegradationCurveData() {
  const raw = readJSON(resolve(resultsDir, 'metrics/degradation_curve_data.json'))
  if (!raw) return { raw: [], binned: [] }

  // Bin boundaries
  const getBin = (n) => {
    if (n === 0) return '0'
    if (n <= 5) return '1-5'
    if (n <= 20) return '6-20'
    return '21+'
  }
  const BIN_ORDER = ['0', '1-5', '6-20', '21+']

  // Group by model + relation + bin
  const groups = {}
  for (const row of raw) {
    const key = `${row.model}|||${row.relation}|||${getBin(row.n_train_edges)}`
    if (!groups[key]) groups[key] = []
    groups[key].push(row.auprc)
  }

  const binned = []
  for (const [key, values] of Object.entries(groups)) {
    const [model, relation, bin] = key.split('|||')
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    binned.push({ model, relation, bin, mean_auprc: parseFloat(mean.toFixed(4)), n: values.length })
  }

  // Sort by model, relation, bin order
  binned.sort((a, b) => {
    if (a.model !== b.model) return a.model.localeCompare(b.model)
    if (a.relation !== b.relation) return a.relation.localeCompare(b.relation)
    return BIN_ORDER.indexOf(a.bin) - BIN_ORDER.indexOf(b.bin)
  })

  // Sample raw to reasonable size (max 500 rows for charting)
  let sampledRaw = raw
  if (raw.length > 500) {
    const step = Math.ceil(raw.length / 500)
    sampledRaw = raw.filter((_, i) => i % step === 0)
  }

  return { raw: sampledRaw, binned }
}

// ---------------------------------------------------------------------------
// Case studies
// ---------------------------------------------------------------------------
function buildCaseStudies() {
  const predA = readCSV(resolve(resultsDir, 'predictions/case_study_caseA_txgnn.csv'))
  const predB = readCSV(resolve(resultsDir, 'predictions/case_study_caseB_txgnn.csv'))
  const pathsA = readCSV(resolve(resultsDir, 'predictions/case_study_caseA_paths_txgnn.csv'))
  const pathsB = readCSV(resolve(resultsDir, 'predictions/case_study_caseB_paths_txgnn.csv'))

  const parseRow = (r) => ({
    disease_id: r.disease_id,
    relation: r.relation,
    rank: parseInt(r.rank),
    drug_id: r.drug_id,
    drug_name: r.drug_name,
    score: parseFloat(r.score),
    is_positive: r.is_positive === 'True' || r.is_positive === 'true',
  })

  const parsePath = (r) => ({
    drug_id: r.drug_id,
    via_entity: r.via_entity,
    via_type: r.via_type,
    relation_drug_to_entity: r.relation_drug_to_entity,
    relation_entity_to_disease: r.relation_entity_to_disease,
    disease_id: r.disease_id,
  })

  // Case A: indication only, top-20
  const caseAInd = predA.filter((r) => r.relation === 'indication').map(parseRow)
  const caseAContra = predA.filter((r) => r.relation === 'contraindication').map(parseRow)

  return {
    caseA: {
      diseaseId: '24573',
      diseaseName: 'Familial Hypertrophic Cardiomyopathy',
      shortName: 'FHC',
      nPos: 1,
      relation: 'indication',
      auprc: 0.025,
      source: 'results/predictions/case_study_caseA_txgnn.csv',
      predictions: caseAInd,
      contraindications: caseAContra,
      paths: pathsA.map(parsePath),
      pathsSource: 'results/predictions/case_study_caseA_paths_txgnn.csv',
      knownDrug: 'Propranolol (DB00571)',
      knownDrugInTop20: false,
      notes:
        'All top-20 scores negative. Propranolol (the one known indication) does not appear in top-20. Model fails on this rare disease.',
    },
    caseB: {
      diseaseId: '5545',
      diseaseName: 'Staphylococcus Aureus Infection',
      shortName: 'S. aureus',
      nPos: 45,
      relation: 'indication',
      auprc: 0.088,
      source: 'results/predictions/case_study_caseB_txgnn.csv',
      predictions: predB.map(parseRow),
      paths: pathsB.map(parsePath),
      pathsSource: 'results/predictions/case_study_caseB_paths_txgnn.csv',
      firstPositiveRank: 18,
      firstPositiveDrug: 'Benzylpenicillin (DB01053)',
      notes:
        'Benzylpenicillin at rank 18 (is_positive=True). Mupirocin (rank 3) and Doxycycline (rank 5) clinically plausible. Three cancer drugs in top-10 (Etoposide, Carboplatin, Bleomycin).',
    },
  }
}

// ---------------------------------------------------------------------------
// Report sections
// ---------------------------------------------------------------------------
function buildReportSections() {
  const sections = [
    { filename: '01_introduction.md', title: 'Introduction' },
    { filename: '02_methods.md', title: 'Methods' },
    { filename: '03_results.md', title: 'Results' },
    { filename: '04_case_studies.md', title: 'Case Studies' },
    { filename: '05_discussion.md', title: 'Discussion' },
  ]
  return sections.map((s) => ({
    ...s,
    content: readText(resolve(reportDir, s.filename)),
    source: `report/sections/${s.filename}`,
  }))
}

// ---------------------------------------------------------------------------
// Assemble and write output
// ---------------------------------------------------------------------------
const output = {
  generatedAt: new Date().toISOString(),
  paperReportedNote:
    'paper_reported values sourced from Huang et al. (2024), Nature Medicine, Supplementary Tables S1 (indication AUPRC) and S2 (contraindication AUPRC), MOESM1 ESM. DOI: 10.1038/s41591-024-03233-x',
  paperReported: PAPER_REPORTED,
  comparisonTable: buildComparisonTable(),
  ablationMatrix: buildAblationMatrix(),
  q6AblationTable: buildQ6AblationTable(),
  degradationCurveData: buildDegradationCurveData(),
  caseStudies: buildCaseStudies(),
  reportSections: buildReportSections(),
}

writeFileSync(outputPath, JSON.stringify(output, null, 2), 'utf-8')
console.log(`[build-data] Written ${outputPath}`)
console.log(`[build-data] comparisonTable rows: ${output.comparisonTable.length}`)
console.log(`[build-data] reportSections: ${output.reportSections.length}`)
console.log(`[build-data] degradationCurve binned rows: ${output.degradationCurveData.binned.length}`)
