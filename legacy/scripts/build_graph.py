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
from typing import Dict, List, Sequence, Tuple, Set
from itertools import combinations
import re
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
    cleaned = [
        p.strip()
        for p in parts
        if p.strip() and p.strip().lower() not in {"unknown", "none"}
    ]
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
                    extra={
                        k: v
                        for k, v in row.items()
                        if k not in {"id", "name", "description", "effect"}
                    },
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


# ---------- text normalization helpers (for matching inside descriptions) ----------
def _normalize_text(s: str | None) -> str:
    if not s:
        return " "
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f" {s} "


def _singularize_last_token(phrase: str) -> str:
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
    v: Set[str] = set()
    base = base.strip()
    if not base:
        return {" "}
    sing = _singularize_last_token(base)
    plur = _pluralize_last_token(sing)
    for form in {base, sing, plur}:
        v.add(f" {form} ")
    return v


# ---------- derived edges: items mention locations ----------
def add_location_mentioned_edges(nodes: List[NodeRecord], edges: List[EdgeRecord]) -> None:
    item_nodes = [n for n in nodes if n.node_type == "item"]
    loc_nodes = [n for n in nodes if n.node_type == "location"]

    variants_by_loc_id: Dict[str, Set[str]] = {}
    for loc in loc_nodes:
        canon = _normalize_text(loc.name).strip()
        canon = canon.strip()
        canon_sing = _singularize_last_token(canon)
        variants_by_loc_id[loc.node_id] = _name_variants(canon_sing)

    seen: Set[Tuple[str, str]] = set()
    for it in item_nodes:
        desc_norm = _normalize_text(it.description or "")
        if len(desc_norm.strip()) == 0:
            continue

        for loc in loc_nodes:
            for variant in variants_by_loc_id.get(loc.node_id, ()):
                if variant in desc_norm:
                    pair = (it.node_id, loc.node_id)
                    if pair not in seen:
                        edges.append(
                            EdgeRecord(
                                source=it.node_id,
                                target=loc.node_id,
                                edge_type="location_mentioned",
                                relationship="location_mentioned",
                                metadata={"matched_variant": variant.strip()},
                            )
                        )
                        seen.add(pair)
                    break


# ---------- derived edges: items mention other items ----------
def add_related_item_edges(nodes: List[NodeRecord], edges: List[EdgeRecord]) -> None:
    item_nodes = [n for n in nodes if n.node_type == "item"]

    variants_by_id: Dict[str, Set[str]] = {}
    for node in item_nodes:
        canon = _normalize_text(node.name).strip()
        canon = canon.strip()
        canon_sing = _singularize_last_token(canon)
        variants_by_id[node.node_id] = _name_variants(canon_sing)

    seen: Set[Tuple[str, str]] = set()
    for src in item_nodes:
        desc_norm = _normalize_text(src.description or "")
        if len(desc_norm.strip()) == 0:
            continue

        for tgt in item_nodes:
            if tgt.node_id == src.node_id:
                continue
            for v in variants_by_id[tgt.node_id]:
                if v in desc_norm:
                    pair = (src.node_id, tgt.node_id)
                    if pair not in seen:
                        edges.append(
                            EdgeRecord(
                                source=src.node_id,
                                target=tgt.node_id,
                                edge_type="related_item",
                                relationship="related_item",
                                metadata={"matched_variant": v.strip()},
                            )
                        )
                        seen.add(pair)
                    break


# ---------- derived edges: share_location between bosses/creatures/npcs ----------
def add_share_location_edges(edges: List[EdgeRecord]) -> None:
    endpoints = ["bosses", "creatures", "npcs"]

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

    pair_weight: Dict[Tuple[str, str], int] = {}
    pair_locs: Dict[Tuple[str, str], Set[str]] = {}

    for loc_key, ids in buckets.items():
        ids_list = sorted(ids)
        if len(ids_list) < 2:
            continue
        for a, b in combinations(ids_list, 2):
            pair = (a, b)
            pair_weight[pair] = pair_weight.get(pair, 0) + 1
            pair_locs.setdefault(pair, set()).add(loc_key)

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


def add_weapon_metadata(nodes: List[NodeRecord]) -> None:
    """Embed weapon-specific metadata directly on the weapon node."""

    def _clean_requirements(raw: List[dict] | None) -> List[dict]:
        cleaned: List[dict] = []
        for entry in raw or []:
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            cleaned.append(
                {
                    "name": name,
                    "amount": entry.get("amount"),
                }
            )
        return cleaned

    def _clean_scales(raw: List[dict] | None) -> List[dict]:
        cleaned: List[dict] = []
        for entry in raw or []:
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            scaling = (entry.get("scaling") or "").strip()
            cleaned.append({"name": name, "scaling": scaling})
        return cleaned

    node_index: Dict[str, NodeRecord] = {node.node_id: node for node in nodes}

    weapon_payload = read_endpoint("weapons")
    for row in weapon_payload:
        weapon_id = row.get("id")
        if not weapon_id:
            continue
        record = node_index.get(weapon_id)
        if not record:
            continue

        extra = record.extra

        category = (row.get("category") or "").strip()
        if category:
            extra["weapon_category"] = category

        req_clean = _clean_requirements(row.get("requiredAttributes"))
        if req_clean:
            extra["weapon_required_attributes"] = req_clean

        scales_clean = _clean_scales(row.get("scalesWith"))
        if scales_clean:
            extra["weapon_scales_with"] = scales_clean


def add_npc_role_edges(nodes: List[NodeRecord], edges: List[EdgeRecord]) -> None:
    """
    Store NPC roles on the NPC node (extra['npc_roles'] = list[str]).
    No role nodes or npc_has_role edges are created.
    """
    node_index: Dict[str, NodeRecord] = {node.node_id: node for node in nodes}

    def _split_roles(raw: str) -> List[str]:
        if not isinstance(raw, str):
            return []
        parts = [raw]
        for sep in [",", " /", "/", " and ", " & "]:
            nxt: List[str] = []
            for p in parts:
                nxt.extend(p.split(sep))
            parts = nxt
        return [p.strip() for p in parts if p and p.strip()]

    npc_payload = read_endpoint("npcs")
    for row in npc_payload:
        npc_id = row.get("id")
        if not npc_id or npc_id not in node_index:
            continue
        roles_raw = row.get("role")
        roles = _split_roles(roles_raw) if roles_raw else []
        if not roles:
            continue
        extra = node_index[npc_id].extra
        existing = set(extra.get("npc_roles", []))
        extra["npc_roles"] = sorted(existing.union(roles))


# ---------- dataframe serialization ----------
def to_dataframe(
    nodes: List[NodeRecord], edges: List[EdgeRecord]
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        add_weapon_metadata(nodes)
        add_npc_role_edges(nodes, edges)
        add_share_location_edges(edges)
        add_related_item_edges(nodes, edges)
        add_location_mentioned_edges(nodes, edges)

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
