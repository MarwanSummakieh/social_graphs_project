## Copilot / AI agent instructions — social_graphs_project

Purpose
- Be productive quickly: this repo investigates the Elden Ring universe by combining network analysis (graphs) with NLP on lore text. The canonical project brief is in `project_guidelines.md` — read it first.

Big picture (what you'll help implement)
- Data acquisition: fetch item/character/location pages from the Elden Ring wiki API (docs: https://docs.eldenring.fanapis.com/).
- Data engineering: clean and normalize text, build node/edge tables and attribute columns suitable for network analysis.
- Analysis: run network-science workflows (degree, centrality, clustering) and link numeric graph features to textual signals (topics, sentiment, embeddings).
- Deliverables: an explainer notebook hosted on GitHub, a <=5-page PDF paper, and a <=1:30 video pitch. See `project_guidelines.md` for exact expectations.

Where to look in this repo
- `project_guidelines.md` — single-source project goals, deliverables, API reference and collaboration expectations.

Project-specific conventions and patterns (discoverable)
- Focus on a combined network + text finding (do not treat the two analyses as independent outputs; show how they connect).
- Datasets should be collected via the Elden Ring wiki API and documented in the notebook's Methods section.
- Host the explainer notebook on GitHub and link it from the paper.

Concrete examples and file names (assumptions noted)
- Expect these artifacts (if missing, create them):
  - `notebooks/explainer_notebook.ipynb` — reproducible analysis and figures.
  - `data/` — raw + cached API responses (git-ignored large files), and `data/README.md` describing sources.
  - `scripts/fetch_data.py` — small script to download and cache API data; include a `--cache` flag.
  - `paper/project_paper.pdf` and `video/project_pitch.mp4` (or links in README).
- Assumption: these names are not required by the repo but align with the project's deliverable expectations in `project_guidelines.md`. Confirm if you use different paths.

How to make changes that match the project's needs
- Reproducibility: every notebook should start with a small cell that documents the data source, date of acquisition, and provides a short snippet to recreate the dataset (or reference `scripts/fetch_data.py`).
- Minimal scripts: prefer small, testable Python scripts (one responsibility per file). Examples: `scripts/graph_builder.py` builds node/edge CSVs from cached API JSON.
- Caching: when calling the public API, cache JSON responses in `data/raw/` and load from cache when present.

Integration points and dependencies
- External: Elden Ring fan API (https://docs.eldenring.fanapis.com/). Handle rate limits and include caching.
- Internal: notebooks and small scripts are the primary integration surface. Keep the code modular so analysis cells import helper functions from `scripts/` rather than copy-pasting logic in many notebook cells.

Agent editing rules (concise)
- Read `project_guidelines.md` before coding. Keep edits focused on reproducible data acquisition, clear graph construction, and one canonical analysis pipeline that links graph features to text-derived features.
- When adding files, update (or create) `README.md` and `data/README.md` to describe how to run the pipeline and where cached data is stored.
- Do not add large raw data files to the repo; add `.gitignore` entries for `data/raw/` and explain how to obtain data via `scripts/fetch_data.py`.

Questions for the maintainers (please answer to refine instructions)
1. Preferred paths/names for notebooks, scripts and data (we guessed `notebooks/`, `scripts/`, `data/`).
2. Any existing code style or Python environment (requirements file, Conda env) to reference?

If something is missing or unclear, ask a short targeted question referencing a filename (e.g., “Where should cached API JSON go — `data/raw/` or `data/cache/`?”).

— end of instructions —
