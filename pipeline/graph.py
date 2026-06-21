"""Build the Elden Ring multimodal graph from raw API payloads.

This module is deliberately explicit about *semantics*:

* Every node carries a typed, defensible **Intelligence<->Faith affinity** derived
  from real stat data (weapon scaling letters, spell stat requirements) rather
  than fragile substring guessing.
* Edges are **typed and kept separate** (``located_in``, ``drops``,
  ``shares_location``, ``mentions``) so downstream analysis can choose which
  relations to combine instead of silently merging incompatible semantics.
* ``mentions`` edges use **word-boundary matching** with name variants, so
  "Rennala" matches "Rennala" but not a substring of another word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

from . import config


# --------------------------------------------------------------------------- #
# Text normalisation helpers
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def norm_key(value: str | None) -> str:
    """Aggressive normalisation for dictionary lookups (drops, locations)."""
    if not value:
        return ""
    value = _PUNCT.sub(" ", value.lower())
    return _WS.sub(" ", value).strip()


def norm_text(value: str | None) -> str:
    """Space-padded normalised text for word-boundary substring matching."""
    key = norm_key(value)
    return f" {key} " if key else " "


def _singularise(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 2 and re.search(r"(ches|shes|xes|ses|zes)$", token):
        return token[:-2]
    if len(token) > 1 and token.endswith("s"):
        return token[:-1]
    return token


def name_variants(name: str) -> set[str]:
    """Space-padded variants used to detect a name inside free text.

    Includes the full name, the pre-comma head (``"Malenia, Blade of
    Miquella" -> "Malenia"``) and a singular form of the last token.
    """
    variants: set[str] = set()
    base = norm_key(name)
    if not base:
        return variants
    candidates = {base}
    if "," in name:
        head = norm_key(name.split(",")[0])
        if head:
            candidates.add(head)
    for cand in list(candidates):
        tokens = cand.split()
        if tokens:
            tokens[-1] = _singularise(tokens[-1])
            candidates.add(" ".join(tokens))
    for cand in candidates:
        if len(cand) >= 3:
            variants.add(f" {cand} ")
    return variants


# --------------------------------------------------------------------------- #
# Node model
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    node_id: str
    name: str
    node_type: str
    text: str = ""               # full lore text (description/effect/quote/role)
    int_affinity: float = 0.0    # 0..1 magnitude of Intelligence scaling/req
    fai_affinity: float = 0.0    # 0..1 magnitude of Faith scaling/req
    faction: str = "Neither"
    category: str | None = None
    region: str | None = None
    roles: list[str] = field(default_factory=list)
    is_merchant: bool = False
    fate: str | None = None
    # transient build-time fields (not all exported verbatim)
    _location_raw: str | None = None
    _drops: list[str] = field(default_factory=list)

    @property
    def axis(self) -> float:
        """Signed Int<->Faith position in [-1, 1] (-1 Int, +1 Faith)."""
        return round(self.fai_affinity - self.int_affinity, 3)


# --------------------------------------------------------------------------- #
# Faction / affinity derivation
# --------------------------------------------------------------------------- #
def _stat_bucket(stat_name: str) -> str | None:
    s = stat_name.strip().lower()
    if s in config.INT_ALIASES:
        return "int"
    if s in config.FAI_ALIASES:
        return "fai"
    return None


def _affinity_from_scaling(scales: list[dict]) -> tuple[float, float]:
    """Weapons/shields: convert scaling letters to Int/Faith magnitudes."""
    i = f = 0.0
    for entry in scales or []:
        bucket = _stat_bucket(entry.get("name", ""))
        if not bucket:
            continue
        letter = (entry.get("scaling") or "").strip().upper()
        mag = config.SCALING_MAGNITUDE.get(letter, 0.0)
        if bucket == "int":
            i = max(i, mag)
        else:
            f = max(f, mag)
    return i, f


def _affinity_from_requires(requires: list[dict]) -> tuple[float, float]:
    """Spells: convert stat requirement amounts to Int/Faith magnitudes.

    Requirements range roughly 0..70; we normalise by 60 and clip to 1.0.
    """
    i = f = 0.0
    for entry in requires or []:
        bucket = _stat_bucket(entry.get("name", ""))
        if not bucket:
            continue
        amount = entry.get("amount") or 0
        try:
            mag = min(float(amount) / 60.0, 1.0)
        except (TypeError, ValueError):
            mag = 0.0
        if bucket == "int":
            i = max(i, mag)
        else:
            f = max(f, mag)
    return i, f


def classify_faction(int_aff: float, fai_aff: float) -> str:
    strong, weak = max(int_aff, fai_aff), min(int_aff, fai_aff)
    if strong < config.FACTION_THRESHOLD:
        return "Neither"
    if weak >= config.FACTION_THRESHOLD and weak / strong >= config.HYBRID_RATIO:
        return "Hybrid"
    return "Intelligence" if int_aff > fai_aff else "Faith"


# --------------------------------------------------------------------------- #
# Node construction
# --------------------------------------------------------------------------- #
def _entity_text(row: dict, node_type: str) -> str:
    parts: list[str] = []
    for key in ("description", "effect"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    # NPCs have no description; their lore lives in quote + role.
    if node_type == "npc":
        for key in ("quote", "role"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
    return _WS.sub(" ", " ".join(parts)).strip()


def _detect_fate(name: str, text: str) -> str | None:
    low_name = name.lower()
    if "bell bearing" in low_name or "remembrance" in low_name:
        return "explicit_fate_item"
    low_text = text.lower()
    if any(k in low_text for k in config.DEATH_KEYWORDS):
        return "implied_death"
    return None


def build_nodes(raw: dict[str, list[dict]]) -> list[Node]:
    nodes: list[Node] = []
    seen_ids: set[str] = set()

    for endpoint, node_type in config.ENDPOINTS.items():
        for row in raw.get(endpoint, []):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            node_id = row.get("id") or f"{node_type}:{norm_key(name)}"
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)

            text = _entity_text(row, node_type)

            int_aff = fai_aff = 0.0
            if node_type in ("weapon", "shield"):
                int_aff, fai_aff = _affinity_from_scaling(row.get("scalesWith"))
            elif node_type in ("sorcery", "incantation"):
                int_aff, fai_aff = _affinity_from_requires(row.get("requires"))

            roles = _split_multi(row.get("role")) if node_type == "npc" else []
            is_merchant = node_type == "npc" and (
                any(h in " ".join(roles).lower() for h in config.MERCHANT_ROLE_HINTS)
                or any(h in text.lower() for h in config.MERCHANT_TEXT_HINTS)
            )

            nodes.append(
                Node(
                    node_id=node_id,
                    name=name,
                    node_type=node_type,
                    text=text,
                    int_affinity=round(int_aff, 3),
                    fai_affinity=round(fai_aff, 3),
                    faction=classify_faction(int_aff, fai_aff),
                    category=(row.get("category") or None),
                    region=(row.get("region") or None),
                    roles=roles,
                    is_merchant=is_merchant,
                    fate=_detect_fate(name, text),
                    _location_raw=row.get("location"),
                    _drops=_clean_drops(row.get("drops")),
                )
            )
    return nodes


def _split_multi(raw: str | None) -> list[str]:
    if not isinstance(raw, str):
        return []
    parts = [raw]
    for sep in (",", "/", " and ", " & ", ";"):
        nxt: list[str] = []
        for p in parts:
            nxt.extend(p.split(sep))
        parts = nxt
    return [p.strip() for p in parts if p.strip()]


def _clean_drops(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for d in raw:
        if not isinstance(d, str):
            continue
        d = d.strip()
        if not d or d.lower() in {"none", "unknown"}:
            continue
        if re.search(r"\d[\d.,]*\s*runes?$", d.lower()):  # "10.000 Runes"
            continue
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Edge construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0


def build_edges(nodes: list[Node]) -> list[Edge]:
    by_id = {n.node_id: n for n in nodes}
    name_to_id: dict[str, str] = {}
    for n in nodes:                       # last writer wins; fine for lookups
        name_to_id[norm_key(n.name)] = n.node_id
    location_ids = {n.node_id for n in nodes if n.node_type == "location"}

    edges: set[Edge] = set()

    # 1. located_in (from `location` and `region` strings) ------------------- #
    for n in nodes:
        place_strings = []
        if n._location_raw:
            place_strings.extend(_split_multi(n._location_raw))
        if n.region:
            place_strings.append(n.region)
        for place in place_strings:
            tid = name_to_id.get(norm_key(place))
            if tid and tid in location_ids and tid != n.node_id:
                edges.add(Edge(n.node_id, tid, "located_in"))

    # 2. drops --------------------------------------------------------------- #
    for n in nodes:
        for drop in n._drops:
            tid = name_to_id.get(norm_key(drop))
            if tid and tid != n.node_id:
                edges.add(Edge(n.node_id, tid, "drops"))

    # 3. shares_location (characters co-located), weighted ------------------- #
    buckets: dict[str, set[str]] = {}
    for n in nodes:
        if n.node_type not in ("boss", "npc", "creature"):
            continue
        for place in _split_multi(n._location_raw or ""):
            key = norm_key(place)
            if key:
                buckets.setdefault(key, set()).add(n.node_id)
    pair_weight: dict[tuple[str, str], int] = {}
    for ids in buckets.values():
        for a, b in combinations(sorted(ids), 2):
            pair_weight[(a, b)] = pair_weight.get((a, b), 0) + 1
    for (a, b), w in pair_weight.items():
        edges.add(Edge(a, b, "shares_location", float(w)))

    # 4. mentions (boundary-aware) ------------------------------------------- #
    targets = [
        n for n in nodes
        if n.node_type in config.MENTION_TARGET_TYPES
        and len(norm_key(n.name)) >= config.MENTION_MIN_NAME_LEN
    ]
    target_variants = [(n.node_id, name_variants(n.name)) for n in targets]
    for n in nodes:
        if not n.text:
            continue
        haystack = norm_text(n.text)
        for tid, variants in target_variants:
            if tid == n.node_id:
                continue
            if any(v in haystack for v in variants):
                edges.add(Edge(n.node_id, tid, "mentions"))

    return sorted(edges, key=lambda e: (e.edge_type, e.source, e.target))
