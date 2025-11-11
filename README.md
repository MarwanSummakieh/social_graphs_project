# Elden Ring Social Graphs

This repository investigates Elden Ring lore by combining graph structure and NLP features harvested from the public Elden Ring fan API.

## Getting Started

1. **Cache the API payloads**
   ```pwsh
   python scripts/fetch_data.py
   ```
   Cached JSON files are written to `data/raw/` (git-ignored). Use `--refresh` to overwrite existing cache files or `--endpoints` to limit the download scope.

2. **Build node and edge tables**
   ```pwsh
   python scripts/build_graph.py
   ```
   The script reads the cached payloads and produces `data/processed/nodes.csv` and `data/processed/edges.csv`, aligning with the network schema discussed in the project brief.

Both scripts expose `--help` flags for additional options.

### Notebook-only workflow

- Open `notebooks/explainer_notebook.ipynb` and execute all cells to perform data download, cleaning, visualization, modeling, and export steps in a single reproducible document.
   - Set the `REFRESH_CACHE` flag in the first code cell if you need to re-fetch the API payloads.
   - The notebook emits `data/processed/node_features.csv`, `data/processed/node_type_report.csv`, and `data/processed/node_type_logreg.joblib` after the final cell runs.

## Data Layout

- `data/raw/`: Cached API responses (`.json`).
- `data/processed/`: Derived CSV assets; currently `nodes.csv` and `edges.csv`.
- `data/README.md`: Expanded documentation of the data lifecycle.

Refer to `project_guidelines.md` for the broader research goals and deliverable expectations.
