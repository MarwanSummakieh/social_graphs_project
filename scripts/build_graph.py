"""Construct the Elden Ring multimodal graph from cached API payloads.

This script expects `scripts/fetch_data.py` to have already cached the
necessary endpoints inside `data/raw/`. It produces two CSV files:

- `data/processed/nodes.csv`
- `data/processed/edges.csv`

Run `python scripts/build_graph.py --help` for options.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

NODE_CONFIG = {
    "bosses": "boss",
    "npcs": "npc",
    "creatures": "creature",
    "locations": "location",
    "items": "item",
    "armors": "armor",
    "shields": "shield",
    "talismans": "talisman",
    "incantations": "incantation",
    "weapons": "weapon",
}

LOCATION_EDGES = {
    "bosses": "boss_located_in",
    "npcs": "npc_located_in",
    "creatures": "creature_located_in",
}

DROP_EDGES = {
    "bosses": "boss_drops",
    "creatures": "creature_drops",
}

DROP_TARGET_TYPES = {"item", "weapon", "armor", "shield", "talisman", "incantation"}


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    node_type: str
    name: str
    description: str | None
    raw_endpoint: str
    extra: dict


@dataclass(frozen=True)
class EdgeRecord:
    source: str
    target: str
    edge_type: str
    relationship: str
    weight: float | None = None
    metadata: dict | None = None


def read_endpoint(endpoint: str) -> Sequence[dict]:
    path = RAW_DIR / f"{endpoint}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache for '{endpoint}'. Did you run scripts/fetch_data.py?"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def normalise_name(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return value.casefold()


def explode_locations(raw_value: str | None) -> List[str]:
    if not raw_value:
        return []
    separators = [",", " /", "/", " and ", " & "]
    parts = [raw_value]
    for sep in separators:
        new_parts: List[str] = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    cleaned = [p.strip() for p in parts if p.strip() and p.strip().lower() not in {"unknown", "none"}]
    return cleaned
def gather_nodes() -> List[NodeRecord]:
    nodes: List[NodeRecord] = []
    for endpoint, node_type in NODE_CONFIG.items():
        payload = read_endpoint(endpoint)
        for row in payload:
            node_id = row.get("id") or f"{endpoint}:{row.get('name')}"
            description = row.get("description") or row.get("effect")
            name = row.get("name", node_id)
            nodes.append(
                NodeRecord(
                    node_id=node_id,
                    node_type=node_type,
                    name=name,
                    description=description,
                    raw_endpoint=endpoint,
                    extra={k: v for k, v in row.items() if k not in {"id", "name", "description", "effect"}},
                )
            )
    return nodes


def build_edges(nodes: List[NodeRecord]) -> List[EdgeRecord]:
    edges: List[EdgeRecord] = []
    location_nodes = [node for node in nodes if node.node_type == "location"]
    location_index = {normalise_name(node.name): node for node in location_nodes}

    for endpoint, relation in LOCATION_EDGES.items():
        payload = read_endpoint(endpoint)
        for row in payload:
            source_id = row.get("id")
            if not source_id:
                continue
            locations = explode_locations(row.get("location"))
            for loc in locations:
                target = location_index.get(normalise_name(loc))
                if not target:
                    continue
                edges.append(
                    EdgeRecord(
                        source=source_id,
                        target=target.node_id,
                        edge_type=relation,
                        relationship="located_in",
                    )
                )

    drop_index = {
        normalise_name(node.name): node
        for node in nodes
        if node.node_type in DROP_TARGET_TYPES
    }

    for endpoint, edge_type in DROP_EDGES.items():
        payload = read_endpoint(endpoint)
        for row in payload:
            source_id = row.get("id")
            if not source_id:
                continue
            for item_name in row.get("drops", []) or []:
                target = drop_index.get(normalise_name(item_name))
                if not target:
                    continue
                edges.append(
                    EdgeRecord(
                        source=source_id,
                        target=target.node_id,
                        edge_type=edge_type,
                        relationship="drops",
                    )
                )

    return edges
def to_dataframe(nodes: List[NodeRecord], edges: List[EdgeRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = [
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "name": node.name,
            "description": node.description,
            "raw_endpoint": node.raw_endpoint,
            "extra_json": json.dumps(node.extra, ensure_ascii=False, sort_keys=True),
        }
        for node in nodes
    ]

    edge_rows = [
        {
            "source": edge.source,
            "target": edge.target,
            "edge_type": edge.edge_type,
            "relationship": edge.relationship,
            "weight": edge.weight,
            "metadata_json": json.dumps(edge.metadata or {}, ensure_ascii=False, sort_keys=True),
        }
        for edge in edges
    ]

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Skip edge construction (useful for debugging cache issues).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    nodes = gather_nodes()
    edges = build_edges(nodes) if not args.no_edges else []

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    node_df, edge_df = to_dataframe(nodes, edges)
    node_path = PROCESSED_DIR / "nodes.csv"
    edge_path = PROCESSED_DIR / "edges.csv"

    node_df.to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)

    print(f"[saved] {node_path} ({len(node_df)} rows)")
    print(f"[saved] {edge_path} ({len(edge_df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
