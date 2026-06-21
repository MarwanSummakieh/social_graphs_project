"""CLI entry point: ``python -m pipeline [--refresh]``."""
from __future__ import annotations

import argparse

from .build import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a fresh download from the API (ignore cached JSON).",
    )
    args = parser.parse_args()
    run(refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
