# Data Directory

Notebook 01 (`notebooks/Marwan's proposal/01_data_collection.ipynb`) is the canonical way to populate these folders.

- `raw/`: Elden Ring Fan API caches (`*.json`) plus `provenance.json`. Set `FORCE_REFRESH=True` in Notebook 01 to replace existing files.
- `processed/`: Outputs from Notebook 02 – at minimum `nodes.csv` and `edges.csv`, along with any downstream features exported by later notebooks.
- `scraped/`: Optional HTML scrape caches produced by `scripts/scrape_wiki.py` (still experimental, ignored by git).

Legacy instructions that reference `scripts/fetch_data.py` or `scripts/build_graph.py` remain valid but are superseded by the notebook workflow.
