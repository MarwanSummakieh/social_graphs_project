"""NLP features computed once, offline.

* **Sentiment** uses VADER (lexicon + rules, tuned for short text) instead of
  TextBlob's movie-review model -- a better fit for terse lore strings.
* **Tokenisation** produces a small per-node bag-of-words (with a curated
  stop-list) so the *frontend* can recompute TF-IDF keywords per community on
  the fly.  TF-IDF here is just term-frequency x inverse-document-frequency, so
  it ports trivially to JavaScript while the heavy cleaning stays in Python.
"""
from __future__ import annotations

import re
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_TOKEN = re.compile(r"[a-z][a-z'-]{2,}")
MAX_TOKENS_PER_NODE = 40

# English + Elden-Ring/game-mechanic stop words.  Deliberately curated: we keep
# lore-bearing nouns (knight, blood, flame) but drop UI/mechanic filler.
ENGLISH_STOP = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "his", "her", "she", "him", "its", "their", "them", "they", "you", "your",
    "who", "whom", "which", "what", "when", "where", "will", "would", "could",
    "should", "have", "has", "had", "but", "not", "all", "any", "can", "may",
    "one", "two", "three", "into", "out", "off", "than", "then", "there", "here",
    "such", "some", "more", "most", "other", "only", "also", "after", "before",
    "once", "upon", "those", "these", "been", "being", "over", "under", "about",
    "yet", "own", "too", "very", "just", "now", "while", "during", "though",
}
GAME_STOP = {
    "elden", "ring", "player", "attack", "damage", "effect", "effects", "increase",
    "increases", "stat", "stats", "attribute", "attributes", "scaling", "scales",
    "weapon", "weapons", "armor", "armour", "shield", "talisman", "incantation",
    "sorcery", "spell", "item", "items", "skill", "ash", "war", "use", "used",
    "using", "equipped", "wield", "wielding", "required", "requires", "require",
    "boost", "guard", "negation", "passive", "deal", "deals", "cast", "casting",
    "fp", "hp", "stamina", "cost", "upgrade", "smithing", "stone", "found",
    "located", "location", "region", "drop", "drops", "dropped", "obtain",
    "reward", "power", "strong", "great", "greatsword", "sword", "blade",
    "type", "set", "made", "part", "wearer", "worn", "grants", "grant",
}
STOPWORDS = ENGLISH_STOP | GAME_STOP

_analyzer = SentimentIntensityAnalyzer()


@lru_cache(maxsize=8192)
def sentiment(text: str) -> float:
    """VADER compound polarity in [-1, 1]; 0.0 for trivially short text."""
    if not text or len(text) < 5:
        return 0.0
    return round(_analyzer.polarity_scores(text)["compound"], 4)


def tokenize(text: str) -> dict[str, int]:
    """Return a capped {token: count} bag for the given text."""
    if not text:
        return {}
    counts: dict[str, int] = {}
    for match in _TOKEN.findall(text.lower()):
        tok = match.strip("'-")
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    if len(counts) <= MAX_TOKENS_PER_NODE:
        return counts
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_TOKENS_PER_NODE]
    return dict(top)
