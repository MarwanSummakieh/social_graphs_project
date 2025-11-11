# Data Directory

This project caches Elden Ring API responses locally to support reproducible graph construction.

- `raw/`: JSON payloads downloaded from https://eldenring.fanapis.com/. Files are keyed by endpoint and parameters. The directory is ignored by git.
- `processed/`: Derived tabular assets (e.g., `nodes.csv`, `edges.csv`, `node_features.csv`, `node_type_report.csv`, `node_type_logreg.joblib`) created by scripts or the notebook in `notebooks/`.

To refresh the cache, run `python scripts/fetch_data.py --refresh`. See the script help text for details.
