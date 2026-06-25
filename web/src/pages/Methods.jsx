import React from 'react'

export default function Methods() {
  return (
    <div className="page">
      <h1>Methods</h1>

      <section className="section" aria-labelledby="data-splits">
        <h2 id="data-splits">Data and Splits</h2>
        <p>
          PrimeKG (Chandak et al., 2023) — downloaded from Harvard Dataverse (
          <a
            href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM"
            target="_blank"
            rel="noopener noreferrer"
          >
            doi:10.7910/DVN/IXA7BM
          </a>
          ). 8,100,498 total edges. 129,375 nodes. License: CC0.
        </p>
        <table>
          <thead>
            <tr>
              <th>Split</th>
              <th>Description</th>
              <th>Test diseases</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Standard</strong></td>
              <td>Train/val/test on (drug, disease) pairs; diseases appear in all sets</td>
              <td>Diseases present in training</td>
            </tr>
            <tr>
              <td><strong>Zero-shot</strong></td>
              <td>Held-out diseases have zero treatment edges in train</td>
              <td>641 diseases, all with zero approved therapies in train</td>
            </tr>
          </tbody>
        </table>
        <p>
          Edges excluded from GNN message passing: <code>anatomy_protein_present</code> (3.03M)
          and <code>drug_drug</code> (2.67M) — excluded due to VRAM constraint. Remaining: ~2.4M
          of 8.1M edges. Documented as scaled_reproduction deviation.
        </p>
      </section>

      <section className="section" aria-labelledby="models">
        <h2 id="models">Models</h2>

        <h3>GNN Baseline (Q1 — KG condition)</h3>
        <p>
          Two-layer HGT encoder. Node features: <code>nn.Embedding</code> per type (Xavier
          uniform). Hidden dim: 64. Scoring: dot product. Phase 1: link prediction pretrain (30
          epochs). Phase 2: fine-tune on therapeutic edges (up to 100 epochs, early stopping
          patience=10).
        </p>

        <h3>GNN No-KG Baseline (Q1 — no-KG condition)</h3>
        <p>
          Same as GNN baseline but <code>num_layers=0</code>. Encode returns raw embeddings with
          no message passing. Both conditions share the same Phase 1 pretrain. The &ldquo;no-KG&rdquo; label
          refers to absence of message passing, not absence of KG-informed pretraining.
        </p>

        <h3>TxGNN (scaled reproduction)</h3>
        <div className="finding-box">
          <p>
            <strong>Architecture clarification:</strong> TxGNN uses the Heterogeneous Graph
            Transformer (HGT) encoder. The word &ldquo;Transformer&rdquo; in HGT refers to its
            multi-head attention aggregation mechanism — it is a <strong>graph neural network
            (GNN)</strong>, not a large language model or text-based Transformer. No language
            models are trained in this study.
          </p>
        </div>
        <p>
          Same HGT encoder as GNN baseline. Added:{' '}
          <code>DiseaseSimilarityModule</code> (cosine similarity projection, k=5 nearest support
          diseases). Two-phase training: Phase 1 = link prediction pretrain; Phase 2 = therapeutic
          task + triplet-style metric-learning loss (<code>sim_loss_weight=0.3</code>).
        </p>
        <div className="warn-box">
          <p>
            <strong>Deviations from Huang et al. (2024):</strong> hidden_dim 512&rarr;64,
            num_layers 3&rarr;2, num_heads 8&rarr;4, node features are learnable embeddings
            (paper uses pre-trained features which are inaccessible). Results are labeled{' '}
            <code>scaled_reproduction</code> throughout.
          </p>
        </div>

        <h3>Alternatives (Q2)</h3>
        <ul>
          <li>
            <strong>SingleStageModel</strong>: KG link prediction + therapeutic task jointly from
            epoch 1. No Phase 1/Phase 2 split.
          </li>
          <li>
            <strong>JointContrastiveModel</strong>: InfoNCE disease similarity + therapeutic task
            jointly.
          </li>
        </ul>

        <h3>Ablations (Q6)</h3>
        <ul>
          <li>
            <strong>TxGNN attn=ON</strong>: full model (HGT attention).
          </li>
          <li>
            <strong>TxGNN attn=OFF</strong>: SAGEConv with mean aggregation instead of HGT
            attention.
          </li>
        </ul>
      </section>

      <section className="section" aria-labelledby="evaluation">
        <h2 id="evaluation">Evaluation</h2>
        <table>
          <tbody>
            <tr>
              <th scope="row">Primary metric</th>
              <td>AUPRC (Area Under Precision-Recall Curve)</td>
            </tr>
            <tr>
              <th scope="row">Secondary metric</th>
              <td>AUROC</td>
            </tr>
            <tr>
              <th scope="row">Protocol</th>
              <td>Random negative sampling (1:5 positive:negative ratio) for all flat evaluations</td>
            </tr>
            <tr>
              <th scope="row">Seeds</th>
              <td>[42, 0, 1] — report mean ± std</td>
            </tr>
            <tr>
              <th scope="row">Table generation</th>
              <td>
                <code>scripts/generate_table.py</code> reads result JSON files. Numbers never
                typed manually.
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="section" aria-labelledby="leakage">
        <h2 id="leakage">Leakage Check</h2>
        <p>
          Zero-shot results are only valid after{' '}
          <code>scripts/run_leakage_check.py</code> passes. The check asserts that no held-out
          test disease appears in any treatment edge in the training split.
        </p>
        <p>
          Output: <code>results/metrics/leakage_check_seed{'{n}'}.json</code>. Status:{' '}
          <strong>PASS</strong> for all seeds [42, 0, 1].
        </p>
      </section>
    </div>
  )
}
