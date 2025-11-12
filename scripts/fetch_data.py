"""Download Elden Ring API resources with on-disk caching.

Usage
-----
python scripts/fetch_data.py --endpoints bosses npcs --refresh

The script stores JSON payloads under data/raw/<endpoint>.json. Subsequent
runs reuse cached files unless --refresh is provided.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence

import requests

BASE_URL = "https://eldenring.fanapis.com/api"
DEFAULT_ENDPOINTS: List[str] = [
    "bosses",
    "creatures",
    "items",
    "locations",
    "npcs",
    "armors",
    "shields",
    "talismans",
    "incantations",
    "weapons",
]
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RATE_LIMIT_SECONDS = 0.25  # keep polite
SESSION = requests.Session()
PAGE_SIZE = 100


def fetch_endpoint(endpoint: str) -> List[dict]:
    """Fetch all pages from a given API endpoint."""
    page = 0
    results: List[dict] = []

    while True:
        params = {"limit": PAGE_SIZE, "page": page}
        url = f"{BASE_URL}/{endpoint}"
        response = SESSION.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Request failed for {endpoint} page {page}: {response.status_code} {response.text}"
            )

        payload = response.json()
        data = payload.get("data", [])
        if not data:
            break

        results.extend(data)
        total = payload.get("total")
        if total is not None and len(results) >= int(total):
            break

        page += 1
        time.sleep(RATE_LIMIT_SECONDS)

    return results


def dump_cache(endpoint: str, rows: List[dict], refresh: bool) -> Path:
    target = CACHE_DIR / f"{endpoint}.json"
    if target.exists() and not refresh:
        raise FileExistsError(
            f"Cache file {target} already exists. Use --refresh to overwrite."
        )

    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return target


def load_cache(endpoint: str) -> List[dict]:
    target = CACHE_DIR / f"{endpoint}.json"
    if not target.exists():
        raise FileNotFoundError(f"Cache file {target} not found.")
    return json.loads(target.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=DEFAULT_ENDPOINTS,
        help="API endpoints to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force download even if cache files already exist.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore on-disk cache and fetch fresh data.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported default endpoints and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.list:
        print("Default endpoints:", ", ".join(DEFAULT_ENDPOINTS))
        return 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for endpoint in args.endpoints:
        cache_path = CACHE_DIR / f"{endpoint}.json"

        if cache_path.exists() and not (args.refresh or args.no_cache):
            print(f"[cache] Using existing {cache_path}")
            continue

        print(f"[fetch] {endpoint}")
        rows = fetch_endpoint(endpoint)
        dump_cache(endpoint, rows, refresh=True)
        print(f"[saved] {cache_path} ({len(rows)} records)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
