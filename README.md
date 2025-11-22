# Elden Ring Social Graphs

**Research focus** – “The Architecture of Fate in Elden Ring.” We study how stat mechanics (Intelligence vs Faith scaling, utility items) dictate the topology of characters, items, and endings.

## Notebook-Centric Workflow

All reproducible steps live in `notebooks/Marwan's proposal/`:

1. `01_data_collection.ipynb` – Download and cache Elden Ring Fan API payloads (items, weapons, NPCs, locations, bosses, armors, talismans, incantations) with on-notebook provenance.
2. `02_graph_construction.ipynb` – Transform raw JSON into `data/processed/nodes.csv` and `data/processed/edges.csv`, tagging Int/Faith scaling and bell-bearing fate markers.
3. `03_network_analysis.ipynb` – Run community detection, assortativity, centrality, and path analyses to evidence the Intelligence/Faith “schism” and “tragic hub” dynamics.
4. `04_nlp_analysis.ipynb` – Perform TF–IDF vocabulary splits and fate-lexicon scoring to link textual lore with mechanics.

Execute the notebooks sequentially (01 → 02 → 03 → 04). Each notebook includes configuration cells so reviewers can refresh caches or tweak parameters without touching standalone scripts.

### Legacy scripts

Historical helpers remain in `scripts/` (`fetch_data.py`, `build_graph.py`, etc.) but are no longer the source of truth. Prefer the notebook pipeline for grading and reproducibility.

## Data Layout

- `data/raw/`: Notebook-managed API caches (JSON) + provenance log.
- `data/processed/`: Outputs from Notebook 02 (`nodes.csv`, `edges.csv`, derived features).
- `data/scraped/`: Optional HTML scrape caches created by `scripts/scrape_wiki.py` (still git-ignored).
- `data/README.md`: Detailed data lifecycle notes.

Consult `project_guidelines.md` for deliverable requirements, then follow the notebook series for the “Bipolar World / Tragedy of Utility / Illusion of Choice” study.
