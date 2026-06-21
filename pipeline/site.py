"""Assemble a single, self-contained ``docs/index.html``.

Every asset (stylesheet, vendored JS libraries, application JS, and the dataset)
is inlined into one HTML file. The result works by double-click, over
``file://``, fully offline, and on GitHub Pages — no server, no CDN, no relative
sub-paths to resolve. This is what makes the page robust to "I just opened it".
"""
from __future__ import annotations

import json
import re

from . import config

DOCS = config.PROJECT_ROOT / "docs"
SRC = DOCS / "_src" / "template.html"
CSS = DOCS / "css" / "style.css"
VENDOR = DOCS / "vendor"
JS = DOCS / "js"

# Load order matters: graphology -> library -> sigma -> chart.
VENDOR_FILES = ["graphology.js", "graphology-library.js", "sigma.js", "chart.js"]


def _safe_inline(js: str) -> str:
    """Prevent a stray ``</script>`` in inlined JS from closing the block."""
    return re.sub(r"</script", r"<\\/script", js, flags=re.IGNORECASE)


def build_site(dataset: dict) -> None:
    template = SRC.read_text(encoding="utf-8")
    style = CSS.read_text(encoding="utf-8")

    libs = "\n".join(_safe_inline((VENDOR / f).read_text(encoding="utf-8")) for f in VENDOR_FILES)

    def js(name: str) -> str:
        return _safe_inline((JS / name).read_text(encoding="utf-8"))

    data_blob = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))
    data_js = _safe_inline("window.ERA_DATA=" + data_blob + ";")

    html = (
        template
        .replace("/*__STYLE__*/", style)
        .replace("/*__LIBS__*/", libs)
        .replace("/*__DATA__*/", data_js)
        .replace("/*__ANALYSIS__*/", js("analysis.js"))
        .replace("/*__APP__*/", js("app.js"))
        .replace("/*__STORY__*/", js("story.js"))
        .replace("/*__DLC__*/", js("dlc.js"))
    )

    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"[saved] {out}  ({out.stat().st_size / 1e6:.2f} MB, self-contained)")
