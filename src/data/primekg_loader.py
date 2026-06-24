"""
Loads the downloaded PrimeKG into PyG HeteroData objects.
Requires data/raw/kg.csv to exist (run scripts/download_primekg.py first).
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# torch and torch_geometric are imported lazily inside build_pyg_heterodata
# to ensure pandas reads the CSV before torch's C extensions modify the allocator.
# (Importing torch_geometric before reading a large CSV causes a Windows access violation.)

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# Relation types treated as therapeutic (drug–disease)
THERAPEUTIC_RELATIONS = {"indication", "contraindication"}


def load_kg_csv(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    kg_path = raw_dir / "kg.csv"
    if not kg_path.exists():
        raise FileNotFoundError(
            f"{kg_path} not found. Run: python scripts/download_primekg.py"
        )
    kg = pd.read_csv(kg_path, low_memory=False)
    print(f"[loader] Loaded kg.csv: {len(kg):,} edges, columns={list(kg.columns)}")
    return kg


def build_entity_index(kg: pd.DataFrame) -> dict[str, dict]:
    """
    Returns a dict: node_type → {original_id: local_int_index}
    Works from x_type/x_id and y_type/y_id columns.
    """
    index: dict[str, dict] = {}
    for col_type, col_id in [("x_type", "x_id"), ("y_type", "y_id")]:
        for node_type, group in kg.groupby(col_type):
            if node_type not in index:
                index[node_type] = {}
            for raw_id in group[col_id].unique():
                if raw_id not in index[node_type]:
                    index[node_type][raw_id] = len(index[node_type])
    return index



# Edge types dropped from GPU message passing to fit within 8 GB VRAM.
# anatomy_protein_present: 3,036,406 edges — not on drug-disease paths
# drug_drug: 2,672,628 edges — drug-drug interactions, not relevant for repurposing
# Removing these reduces total edges from 8.1M to ~2.4M (fits in 8 GB VRAM).
# Documented as scaled_reproduction constraint in CONTEXT.md.
SKIP_RELATIONS_DEFAULT = frozenset({"anatomy_protein_present", "drug_drug"})


def build_pyg_heterodata(
    kg: pd.DataFrame,
    entity_index: dict,
    skip_relations: frozenset = SKIP_RELATIONS_DEFAULT,
):
    """
    Converts the edge dataframe into a PyG HeteroData object.
    Each unique (x_type, relation, y_type) triple becomes one edge type.

    skip_relations: relation names to exclude from message passing.
    Default drops anatomy_protein_present and drug_drug (too large for 8 GB VRAM).
    """
    import torch  # deferred: must import AFTER pandas has read any large CSV
    from torch_geometric.data import HeteroData  # deferred: same reason

    data = HeteroData()

    # Node feature placeholders (identity index; real features loaded separately)
    for node_type, id_map in entity_index.items():
        n = len(id_map)
        data[node_type].num_nodes = n
        data[node_type].node_id = torch.arange(n)

    skipped = []
    for (x_type, relation, y_type), group in kg.groupby(["x_type", "relation", "y_type"]):
        if relation in skip_relations:
            skipped.append(relation)
            continue
        x_idx = torch.tensor(
            [entity_index[x_type][i] for i in group["x_id"]], dtype=torch.long
        )
        y_idx = torch.tensor(
            [entity_index[y_type][i] for i in group["y_id"]], dtype=torch.long
        )
        data[x_type, relation, y_type].edge_index = torch.stack([x_idx, y_idx], dim=0)

    if skipped:
        print(f"[loader] Skipped {len(set(skipped))} large relation type(s) "
              f"({', '.join(sorted(set(skipped)))}) to fit 8 GB VRAM.")

    return data


def get_therapeutic_edges(kg: pd.DataFrame) -> pd.DataFrame:
    """Returns only indication and contraindication edges."""
    return kg[kg["relation"].isin(THERAPEUTIC_RELATIONS)].copy()


def load_primekg(raw_dir: Path = RAW_DIR):
    """Main entry point. Returns (kg_df, entity_index, heterodata)."""
    kg = load_kg_csv(raw_dir)
    entity_index = build_entity_index(kg)
    heterodata = build_pyg_heterodata(kg, entity_index)
    print(f"[loader] Node types: {list(entity_index.keys())}")
    print(f"[loader] Total node types: {len(entity_index)}")
    print(f"[loader] Edge types: {len(heterodata.edge_types)}")
    return kg, entity_index, heterodata
