"""Download Elden Ring Fan API resources with on-disk caching.

The API is paginated (``limit``/``page``) and rate-limited politely.  Responses
are cached as JSON under ``data/raw/<endpoint>.json`` and reused unless
``refresh=True`` is passed.
"""
from __future__ import annotations

import json
import time
from typing import Iterable

import requests

from . import config


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "social-graphs-eldenring/2.0 (academic)"})
    return s


def fetch_endpoint(endpoint: str, session: requests.Session) -> list[dict]:
    """Fetch every page of a single endpoint."""
    page, rows = 0, []
    while True:
        resp = session.get(
            f"{config.API_BASE_URL}/{endpoint}",
            params={"limit": config.PAGE_SIZE, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        if not data:
            break
        rows.extend(data)
        total = payload.get("total")
        if total is not None and len(rows) >= int(total):
            break
        page += 1
        time.sleep(config.RATE_LIMIT_SECONDS)
    return rows


def load_or_fetch(
    endpoints: Iterable[str] | None = None, *, refresh: bool = False
) -> dict[str, list[dict]]:
    """Return ``{endpoint: rows}`` for the requested endpoints.

    Cached files are reused unless ``refresh`` is set.
    """
    endpoints = list(endpoints or config.ENDPOINTS.keys())
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()
    out: dict[str, list[dict]] = {}

    for endpoint in endpoints:
        cache = config.RAW_DIR / f"{endpoint}.json"
        if cache.exists() and not refresh:
            out[endpoint] = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[cache] {endpoint}: {len(out[endpoint])} rows")
            continue
        print(f"[fetch] {endpoint} ...")
        rows = fetch_endpoint(endpoint, session)
        cache.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        out[endpoint] = rows
        print(f"[saved] {endpoint}: {len(rows)} rows -> {cache.name}")
    return out
