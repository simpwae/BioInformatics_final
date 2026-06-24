"""
Downloads PrimeKG from Harvard Dataverse and writes data/primekg_stats.json
from the actual downloaded data — not from any paper or blog.

Files downloaded:
  data/raw/kg.csv          — full edge list (source of truth)
  data/raw/nodes.tab       — node metadata
  data/raw/disease_features.tab
  data/raw/drug_features.tab

Stats written to:
  data/primekg_stats.json  — computed from the files, not copied from memory
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
STATS_OUT = ROOT / "data" / "primekg_stats.json"

DATAVERSE_FILES = {
    "kg.csv":                  "https://dataverse.harvard.edu/api/access/datafile/6180620",
    "nodes.tab":               "https://dataverse.harvard.edu/api/access/datafile/6180617",
    "disease_features.tab":    "https://dataverse.harvard.edu/api/access/datafile/6180618",
    "drug_features.tab":       "https://dataverse.harvard.edu/api/access/datafile/6180619",
}


def _download(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[download] {dest.name} <- {url}")
    tmp = dest.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            t0 = time.time()
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded / total * 100
                    speed = downloaded / (time.time() - t0 + 1e-6) / 1e6
                    print(f"\r  {pct:.1f}%  {downloaded/1e6:.1f}/{total/1e6:.1f} MB  {speed:.1f} MB/s",
                          end="", flush=True)
        tmp.rename(dest)
        print(f"\n  → saved {dest.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def compute_stats(raw_dir: Path) -> dict:
    kg_path = raw_dir / "kg.csv"
    nodes_path = raw_dir / "nodes.tab"

    print("\n[stats] Loading kg.csv ...")
    kg = pd.read_csv(kg_path, low_memory=False)
    print(f"  kg.csv rows: {len(kg):,}")

    stats = {
        "source": "computed from downloaded files",
        "kg_csv_rows": len(kg),
        "kg_csv_columns": list(kg.columns),
    }

    # Edge counts by relation type
    if "relation" in kg.columns:
        rel_counts = kg["relation"].value_counts().to_dict()
        stats["edge_counts_by_relation"] = {k: int(v) for k, v in rel_counts.items()}
        stats["total_edges"] = int(len(kg))
        stats["n_relation_types"] = len(rel_counts)
    else:
        print(f"  [warn] No 'relation' column. Columns: {list(kg.columns)}")

    # Node type counts from kg edges
    if "x_type" in kg.columns and "y_type" in kg.columns:
        x_types = kg["x_type"].value_counts().to_dict()
        y_types = kg["y_type"].value_counts().to_dict()
        # unique nodes per type
        unique_nodes_x = kg.groupby("x_type")["x_id"].nunique().to_dict() if "x_id" in kg.columns else {}
        unique_nodes_y = kg.groupby("y_type")["y_id"].nunique().to_dict() if "y_id" in kg.columns else {}
        # merge
        all_types = set(unique_nodes_x) | set(unique_nodes_y)
        node_counts = {}
        for t in all_types:
            ids_x = set(kg.loc[kg["x_type"] == t, "x_id"].unique()) if "x_id" in kg.columns else set()
            ids_y = set(kg.loc[kg["y_type"] == t, "y_id"].unique()) if "y_id" in kg.columns else set()
            node_counts[t] = len(ids_x | ids_y)
        stats["node_counts_by_type"] = {k: int(v) for k, v in sorted(node_counts.items(), key=lambda x: -x[1])}
        stats["total_nodes"] = int(sum(node_counts.values()))

    # Nodes file
    if nodes_path.exists():
        print("[stats] Loading nodes.tab ...")
        nodes = pd.read_csv(nodes_path, sep="\t", low_memory=False)
        stats["nodes_tab_rows"] = int(len(nodes))
        stats["nodes_tab_columns"] = list(nodes.columns)
        if "node_type" in nodes.columns:
            type_counts = nodes["node_type"].value_counts().to_dict()
            stats["node_counts_from_nodes_tab"] = {k: int(v) for k, v in type_counts.items()}
            stats["total_nodes_from_nodes_tab"] = int(len(nodes))

    # Drug / disease counts from edges
    if "x_type" in kg.columns and "x_id" in kg.columns:
        disease_ids = set()
        drug_ids = set()
        for col_type, col_id in [("x_type", "x_id"), ("y_type", "y_id")]:
            if col_type in kg.columns and col_id in kg.columns:
                disease_ids |= set(kg.loc[kg[col_type] == "disease", col_id].unique())
                drug_ids |= set(kg.loc[kg[col_type].isin(["drug", "compound"]), col_id].unique())
        stats["n_diseases_in_edges"] = len(disease_ids)
        stats["n_drugs_in_edges"] = len(drug_ids)

    return stats


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=== PrimeKG Download ===")
    for fname, url in DATAVERSE_FILES.items():
        _download(url, RAW_DIR / fname)

    print("\n=== Computing stats from downloaded data ===")
    stats = compute_stats(RAW_DIR)
    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[done] Stats written to {STATS_OUT}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
