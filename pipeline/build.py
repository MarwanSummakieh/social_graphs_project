"""Orchestrate the offline pipeline and emit the static dataset.

    fetch  ->  build nodes/edges  ->  NLP features  ->  layout  ->  graph.json

The output (``docs/data/graph.json``) is the *only* artefact the GitHub Pages
frontend needs; all interactive analysis (community detection, centrality,
TF-IDF) is recomputed in the browser from this file.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter

import networkx as nx

from . import config, dlc, fetch, site, text
from .graph import Edge, Node, build_edges, build_nodes


def _layout(nodes: list[Node], edges: list[Edge]) -> dict[str, tuple[float, float]]:
    """Force-directed coordinates over the weighted union of all edges."""
    g = nx.Graph()
    g.add_nodes_from(n.node_id for n in nodes)
    for e in edges:
        # collapse multi-type parallels; keep the strongest weight
        if g.has_edge(e.source, e.target):
            g[e.source][e.target]["weight"] = max(
                g[e.source][e.target]["weight"], e.weight
            )
        else:
            g.add_edge(e.source, e.target, weight=e.weight)
    k = 1.0 / max(len(nodes) ** 0.5, 1.0)
    pos = nx.spring_layout(g, k=k, iterations=60, seed=42, weight="weight")
    return {nid: (round(float(x) * 100, 2), round(float(y) * 100, 2))
            for nid, (x, y) in pos.items()}


def _node_payload(n: Node, pos: tuple[float, float]) -> dict:
    return {
        "id": n.node_id,
        "name": n.name,
        "type": n.node_type,
        "faction": n.faction,
        "ia": n.int_affinity,
        "fa": n.fai_affinity,
        "axis": n.axis,
        "cat": n.category,
        "region": n.region,
        "roles": n.roles,
        "merchant": n.is_merchant,
        "fate": n.fate,
        "sent": text.sentiment(n.text),
        "snippet": n.text[:240],
        "lore": n.text[:1000],          # fuller text for the storytelling dossiers
        "x": pos[0],
        "y": pos[1],
        "tok": text.tokenize(n.text),
    }


def run(*, refresh: bool = False) -> dict:
    raw = fetch.load_or_fetch(refresh=refresh)
    nodes = build_nodes(raw)
    edges = build_edges(nodes)

    print(f"\nNodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print("  by type:", dict(Counter(e.edge_type for e in edges)))
    print("  factions:", dict(Counter(n.faction for n in nodes)))

    pos = _layout(nodes, edges)

    dataset = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": config.API_SOURCE_LABEL,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": dict(Counter(n.node_type for n in nodes)),
            "edge_types": dict(Counter(e.edge_type for e in edges)),
            "factions": dict(Counter(n.faction for n in nodes)),
        },
        "nodes": [_node_payload(n, pos[n.node_id]) for n in nodes],
        "edges": [
            {"s": e.source, "t": e.target, "type": e.edge_type, "w": e.weight}
            for e in edges
        ],
        "dlc": dlc.build_dlc(),
    }

    config.WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.WEB_DATA_DIR / "graph.json"
    out_path.write_text(
        json.dumps(dataset, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[saved] {out_path}  ({size_mb:.2f} MB)")

    # Bake the self-contained single-file site so it works by double-click.
    site.build_site(dataset)
    return dataset
