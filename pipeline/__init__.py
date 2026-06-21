"""Elden Ring social-graphs pipeline.

A small, correct, reproducible pipeline that turns the Elden Ring Fan API into a
single ``docs/data/graph.json`` consumed by the interactive GitHub Pages site.

Run with::

    python -m pipeline            # fetch (cached) + build
    python -m pipeline --refresh  # force a fresh API download
"""
