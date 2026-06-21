"""Central configuration for the Elden Ring network/NLP pipeline.

Everything that the offline pipeline needs to know about *where* data lives and
*how* entities are interpreted is collected here so the rest of the package stays
declarative and testable.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# The frontend is served from /docs (GitHub Pages "main / docs"), so the
# generated dataset is written straight into the site so it can be committed.
WEB_DATA_DIR = PROJECT_ROOT / "docs" / "data"

API_BASE_URL = "https://eldenring.fanapis.com/api"
API_SOURCE_LABEL = "eldenring.fanapis.com"
PAGE_SIZE = 100
RATE_LIMIT_SECONDS = 0.2

# --------------------------------------------------------------------------- #
# Endpoints -> node type.  Order matters only for deterministic output.
# --------------------------------------------------------------------------- #
ENDPOINTS: dict[str, str] = {
    "weapons": "weapon",
    "shields": "shield",
    "sorceries": "sorcery",
    "incantations": "incantation",
    "ashes": "ash_of_war",
    "talismans": "talisman",
    "armors": "armor",
    "items": "item",
    "npcs": "npc",
    "bosses": "boss",
    "creatures": "creature",
    "locations": "location",
}

# Node types whose names are meaningful targets for "mentions" edges.  Matching
# every node against every description is noisy; restricting *targets* to named
# narrative entities keeps the mention graph interpretable.
MENTION_TARGET_TYPES = {"npc", "boss", "creature", "location"}

# Minimum target-name length for a mention to count (guards against short,
# ambiguous names colliding with common words).
MENTION_MIN_NAME_LEN = 5

# --------------------------------------------------------------------------- #
# The Intelligence <-> Faith axis (the project's central thesis).
# Scaling letters map to a magnitude; stat-requirement amounts are normalised
# separately.  Both feed a single signed affinity in [-1, 1].
# --------------------------------------------------------------------------- #
SCALING_MAGNITUDE = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2, "E": 0.1}

# Canonical stat-name spellings as they appear across endpoints.
INT_ALIASES = {"int", "intelligence"}
FAI_ALIASES = {"fai", "fth", "faith"}

# A spell/weapon counts as belonging to a faction only if its affinity exceeds
# this; below it the entity is "Neither".  Hybrids have both sides strong.
FACTION_THRESHOLD = 0.12
HYBRID_RATIO = 0.55  # weaker side >= 55% of stronger side -> Hybrid

# --------------------------------------------------------------------------- #
# Fate markers (the "tragedy of utility" thread).
# --------------------------------------------------------------------------- #
DEATH_KEYWORDS = [
    "died", "death", "slain", "killed", "corpse", "remains", "perished",
    "last words", "grave", "tomb", "deceased", "fallen", "mournful",
]
MERCHANT_ROLE_HINTS = ["shop", "merchant", "nomadic", "goods", "vendor"]
MERCHANT_TEXT_HINTS = ["bell bearing", "merchant", "sells", "wares"]
