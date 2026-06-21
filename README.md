# The Architecture of Fate — Elden Ring Network & NLP Explorer

[![Open the live site](https://img.shields.io/badge/▶_Open_the_live_explorer-GitHub_Pages-c9a86a?style=for-the-badge&logo=github)](https://marwansummakieh.github.io/social_graphs_project/)

**Live site:** https://marwansummakieh.github.io/social_graphs_project/

A network-science + NLP study of the Elden Ring world, with an **interactive web
frontend** hosted on GitHub Pages. The central thesis: game mechanics —
especially the **Intelligence ↔ Faith** stat schism — shape the narrative
topology of items, characters, and places.

> **Live analysis, in the browser.** Toggle which relationship types build the
> graph, filter node types, recolor by faction / community / sentiment, and move
> the Louvain-resolution slider — community detection, centrality, and TF-IDF
> keyword extraction all **recompute live** from a single pre-built dataset.

The site has three views (tabs):

1. **The Network** — the interactive graph explorer described above.
2. **The Story** — *storytelling through data*: pick a character/place and the app
   reconstructs its tale from the game's own item/location/boss text (the "echoes"
   that name it), detects lore events (the Shattering, Night of the Black Knives…),
   and surfaces its bonds — all ranked by how much the world actually speaks of it.
3. **Realm of Shadow** — a curated, motive-focused map of the *Shadow of the
   Erdtree* DLC (the fan API has no DLC data), explaining each figure's motives and
   how they reframe the base game.

The whole site is built into a **single self-contained `docs/index.html`** (CSS,
JS libraries, app code, and data all inlined), so it works by **double-clicking
the file** — offline, no server — as well as on GitHub Pages.

---

## Repository layout

```
pipeline/            # Correct, reproducible Python pipeline (the analysis engine)
  fetch.py           #   paginated, cached API download
  graph.py           #   nodes + Int/Faith affinity + typed, boundary-aware edges
  text.py            #   VADER sentiment + tokenisation for client-side TF-IDF
  build.py           #   orchestration + force-directed layout + JSON export
  __main__.py        #   `python -m pipeline`
docs/                # Static site (GitHub Pages root)
  index.html
  css/style.css
  js/analysis.js     #   graph algorithms (Louvain, centrality, TF-IDF) — DOM-free
  js/app.js          #   Sigma rendering, charts, controls
  data/graph.json    #   committed dataset snapshot (so the site runs out of the box)
requirements.txt     # pipeline deps only (the frontend uses CDN libraries)
legacy/notebooks…    # earlier notebook drafts — superseded by this pipeline
```

## What makes this version "correct"

The earlier notebooks had several defects; this rewrite fixes them:

| Problem in the old code | Fix here |
| --- | --- |
| Faction tagging by fragile exact-string matching (`'int' in names`) | A signed **affinity axis** derived from real `scalesWith` letters and spell `requires` amounts ([`graph.py`](pipeline/graph.py)). Sorceries → Intelligence, incantations → Faith, staves/seals correctly classified. |
| `mentions` edges via unbounded substring matching (false positives) | **Word-boundary** matching with singular/plural + pre-comma name variants; mention *targets* restricted to named entities. 475 clean mention edges instead of tens of thousands of spurious ones. |
| Heterogeneous edges silently merged before community detection | Edges are **typed and kept separate**; the UI lets you choose which relations feed the graph. |
| TextBlob sentiment (movie-review model) on terse lore | **VADER** (lexicon+rules), better suited to short text. |
| Stochastic, non-reproducible Louvain | Seeded RNG → identical communities for identical parameters. |
| Missing dependencies, a corrupted cell, an embedded full-site scrape | A small, declarative package with a complete `requirements.txt`. |

## Run the pipeline (regenerate the dataset)

```bash
pip install -r requirements.txt
python -m pipeline            # uses cached API data if present
python -m pipeline --refresh  # force a fresh download from the API
```

This writes `docs/data/graph.json` (≈1.4 MB: ~2,200 nodes, ~2,000 typed edges,
per-node Int/Faith affinity, VADER sentiment, token bags, and force-directed
coordinates). Raw API responses are cached under `data/raw/` (git-ignored).

## Run the frontend locally

It's a static site — any static server works:

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000
```

## Deploy to GitHub Pages

1. Commit the repo (including `docs/data/graph.json`).
2. Repo **Settings → Pages → Build and deployment → Source: "Deploy from a
   branch"**, branch `main`, folder **`/docs`**.
3. The site publishes at `https://<user>.github.io/<repo>/`.

No build step and no server are required; all interactivity runs client-side
(graphology + sigma.js + Chart.js, loaded from jsDelivr).

## Data source

Elden Ring Fan API — <https://docs.eldenring.fanapis.com/>. Endpoints: weapons,
shields, sorceries, incantations, ashes of war, talismans, armor, items, npcs,
bosses, creatures, locations.
