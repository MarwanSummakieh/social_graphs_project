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
from typing import Dict, List, Sequence, Tuple, Set, DefaultDict
from itertools import combinations
import re
from collections import defaultdict
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

NODE_CONFIG = {
    "bosses": "boss",
    "npcs": "npc",
    "creatures": "creature",
    "locations": "location",
    "items": "item",
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

DROP_TARGET_TYPES = {"item", "weapon"}

DERIVED_ENDPOINT = "derived"

ATTRIBUTE_LABELS = {
    "str": "Strength",
    "dex": "Dexterity",
    "int": "Intelligence",
    "fai": "Faith",
    "arc": "Arcane",
    "end": "Endurance",
    "vit": "Vitality",
    "vig": "Vigor",
    "mind": "Mind",
}


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


def ensure_node_record(
    nodes: List[NodeRecord],
    node_index: Dict[str, NodeRecord],
    node_id: str,
    *,
    node_type: str,
    name: str,
    description: str | None = None,
    extra: dict | None = None,
) -> NodeRecord:
    if node_id in node_index:
        return node_index[node_id]
    record = NodeRecord(
        node_id=node_id,
        node_type=node_type,
        name=name,
        description=description,
        raw_endpoint=DERIVED_ENDPOINT,
        extra=extra or {},
    )
    nodes.append(record)
    node_index[node_id] = record
    return record


def ensure_attribute_node(
    nodes: List[NodeRecord], node_index: Dict[str, NodeRecord], code: str
) -> NodeRecord:
    label = ATTRIBUTE_LABELS.get(code.casefold(), code)
    node_id = f"attribute:{code.casefold()}"
    description = f"Character attribute {label}"
    return ensure_node_record(
        nodes,
        node_index,
        node_id,
        node_type="attribute",
        name=label,
        description=description,
        extra={"code": code.upper()},
    )


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

def _normalize_text(s: str | None) -> str:
    """
    Normalize text for matching: lowercase, strip punctuation except spaces,
    collapse whitespace. Returns a string with single leading/trailing spaces
    so we can safely do ' in ' containment checks with word boundaries.
    """
    if not s:
        return " "
    s = s.lower()
    # keep letters/digits/spaces, turn other chars into spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f" {s} "

def _singularize_last_token(phrase: str) -> str:
    """
    Naive singularizer for the LAST token of a multi-word phrase.
    Handles common English plural rules enough for item names:
      - 'ies' -> 'y'  (e.g., 'bodies' -> 'body')
      - 'es' for sibilants -> drop 'es' (e.g., 'ashes' -> 'ash')
      - trailing 's' -> drop 's'
    Falls back to the original if no rule applies.
    """
    tokens = phrase.split()
    if not tokens:
        return phrase
    w = tokens[-1]
    if len(w) > 3 and w.endswith("ies"):
        w2 = w[:-3] + "y"
    elif len(w) > 2 and re.search(r"(ches|shes|xes|ses|zes)$", w):
        w2 = w[:-2]
    elif len(w) > 1 and w.endswith("s"):
        w2 = w[:-1]
    else:
        w2 = w
    tokens[-1] = w2
    return " ".join(tokens)

def _pluralize_last_token(phrase: str) -> str:
    """
    Simple pluralizer for the LAST token of a phrase to generate a variant:
      - 'y' -> 'ies'
      - sibilants -> + 'es'
      - else -> + 's'
    """
    tokens = phrase.split()
    if not tokens:
        return phrase
    w = tokens[-1]
    if len(w) > 1 and w.endswith("y"):
        w2 = w[:-1] + "ies"
    elif re.search(r"(ch|sh|x|s|z)$", w):
        w2 = w + "es"
    else:
        w2 = w + "s"
    tokens[-1] = w2
    return " ".join(tokens)

def _name_variants(base: str) -> Set[str]:
    """
    Given a normalized item name (lowercase, no punctuation, single spaces),
    return a set of variants we’ll search for in descriptions:
      - base (singular form)
      - plural of base
      - singularized form (in case the canonical name is plural in data)
    All variants are returned padded with leading/trailing spaces for
    safe ' in ' containment checks.
    """
    v: Set[str] = set()
    base = base.strip()
    if not base:
        return {" "}
    sing = _singularize_last_token(base)
    plur = _pluralize_last_token(sing)
    for form in {base, sing, plur}:
        v.add(f" {form} ")
    return v

def add_related_item_edges(nodes: List[NodeRecord], edges: List[EdgeRecord]) -> None:
    """
    For each item A, if A.description mentions the name of item B (case-insensitive,
    simple plural/singular normalization), add a directed edge:
        A --[related_item]--> B
    Avoid self-edges, and only add one edge per (A,B) even if multiple mentions.
    """
    # 1) Collect all item nodes and build normalized-name variant index
    item_nodes = [n for n in nodes if n.node_type == "item"]
    # Map normalized singular base -> node_id (assume uniqueness of canonical names)
    id_by_canonical: Dict[str, str] = {}
    variants_by_id: Dict[str, Set[str]] = {}
    
    for node in item_nodes:
        canon = _normalize_text(node.name).strip()
        # remove outer spaces introduced by _normalize_text()
        canon = canon.strip()
        # force singular canonical key for indexing
        canon_sing = _singularize_last_token(canon)
        id_by_canonical[canon_sing] = node.node_id
        variants_by_id[node.node_id] = _name_variants(canon_sing)

    # 2) For each item, scan its normalized description and look for OTHER item variants
    seen: Set[Tuple[str, str]] = set()  # (source_id, target_id)
    for src in item_nodes:
        desc_norm = _normalize_text(src.description or "")
        if len(desc_norm.strip()) == 0:
            continue

        for tgt in item_nodes:
            if tgt.node_id == src.node_id:
                continue
            for v in variants_by_id[tgt.node_id]:
                # simple, fast whole-phrase containment check with space padding
                if v in desc_norm:
                    pair = (src.node_id, tgt.node_id)
                    if pair not in seen:
                        edges.append(
                            EdgeRecord(
                                source=src.node_id,
                                target=tgt.node_id,
                                edge_type="related_item",
                                relationship="related_item",
                                # optional: weight = 1.0; you could count occurrences if desired
                                metadata={"matched_variant": v.strip()}
                            )
                        )
                        seen.add(pair)
                    break  # don’t add duplicates for multiple variants of same target

def add_share_location_edges(edges: List[EdgeRecord]) -> None:
    """
    Create share_location edges between any pair of entities (bosses, creatures, npcs)
    that appear in the same location. Edge weight is the number of shared locations.
    """
    # endpoints that carry 'location'
    endpoints = ["bosses", "creatures", "npcs"]

    # Build buckets: normalized_location -> set of entity ids that have this location
    buckets: Dict[str, set] = {}
    for endpoint in endpoints:
        payload = read_endpoint(endpoint)
        for row in payload:
            src = row.get("id")
            if not src:
                continue
            for raw_loc in explode_locations(row.get("location")):
                key = normalise_name(raw_loc)
                if not key:
                    continue
                buckets.setdefault(key, set()).add(src)

    # Count co-location pairs and collect which locations they share
    pair_weight: Dict[tuple, int] = {}
    pair_locs: Dict[tuple, set] = {}

    for loc_key, ids in buckets.items():
        ids_list = sorted(ids)  # stable ordering for canonical pair keys
        if len(ids_list) < 2:
            continue
        for a, b in combinations(ids_list, 2):
            pair = (a, b)  # canonical (sorted) pair
            pair_weight[pair] = pair_weight.get(pair, 0) + 1
            pair_locs.setdefault(pair, set()).add(loc_key)

    # Materialize edges once per pair, with weight and metadata
    for (a, b), w in pair_weight.items():
        edges.append(
            EdgeRecord(
                source=a,
                target=b,
                edge_type="share_location",
                relationship="share_location",
                weight=float(w),
                metadata={"locations": sorted(pair_locs[(a, b)])},
            )
        )


def add_weapon_metadata_edges(nodes: List[NodeRecord], edges: List[EdgeRecord]) -> None:
    node_index: Dict[str, NodeRecord] = {node.node_id: node for node in nodes}

    weapon_payload = read_endpoint("weapons")
    for row in weapon_payload:
        weapon_id = row.get("id")
        if not weapon_id or weapon_id not in node_index:
            continue

        category = (row.get("category") or "").strip()
        if category:
            category_id = f"weapon_category:{category.casefold()}"
            category_node = ensure_node_record(
                nodes,
                node_index,
                category_id,
                node_type="weapon_category",
                name=category,
                description=f"Weapon category {category}",
                extra={"category": category},
            )
            edges.append(
                EdgeRecord(
                    source=weapon_id,
                    target=category_node.node_id,
                    edge_type="weapon_category",
                    relationship="belongs_to",
                )
            )

        for requirement in row.get("requiredAttributes") or []:
            attr_name = requirement.get("name")
            if not attr_name:
                continue
            attr_node = ensure_attribute_node(nodes, node_index, attr_name)
            amount = requirement.get("amount")
            metadata = {"amount": amount} if amount is not None else None
            weight = float(amount) if isinstance(amount, (int, float)) else None
            edges.append(
                EdgeRecord(
                    source=weapon_id,
                    target=attr_node.node_id,
                    edge_type="weapon_requires_attribute",
                    relationship="requires",
                    weight=weight,
                    metadata=metadata,
                )
            )

        for scale in row.get("scalesWith") or []:
            attr_name = scale.get("name")
            if not attr_name:
                continue
            attr_node = ensure_attribute_node(nodes, node_index, attr_name)
            scaling = scale.get("scaling")
            metadata = {"scaling": scaling} if scaling else None
            edges.append(
                EdgeRecord(
                    source=weapon_id,
                    target=attr_node.node_id,
                    edge_type="weapon_scales_with",
                    relationship="scales_with",
                    metadata=metadata,
                )
            )


def add_npc_role_edges(nodes: List[NodeRecord], edges: List[EdgeRecord]) -> None:
    node_index: Dict[str, NodeRecord] = {node.node_id: node for node in nodes}

    npc_payload = read_endpoint("npcs")
    for row in npc_payload:
        npc_id = row.get("id")
        if not npc_id or npc_id not in node_index:
            continue
        role_value = row.get("role")
        if not role_value:
            continue
        role_clean = role_value.strip()
        if not role_clean:
            continue
        role_id = f"npc_role:{role_clean.casefold()}"
        role_node = ensure_node_record(
            nodes,
            node_index,
            role_id,
            node_type="npc_role",
            name=role_clean,
            description=f"NPC role: {role_clean}",
        )
        edges.append(
            EdgeRecord(
                source=npc_id,
                target=role_node.node_id,
                edge_type="npc_has_role",
                relationship="has_role",
            )
        )


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
    if not args.no_edges:
        add_weapon_metadata_edges(nodes, edges)
        add_npc_role_edges(nodes, edges)
        add_share_location_edges(edges)
        add_related_item_edges(nodes, edges)

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
