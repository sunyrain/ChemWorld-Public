"""Build the project website at / and its documentation at /docs/."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

from build_explorer_site import build as build_explorer

ROOT = Path(__file__).resolve().parents[1]


def redirect(page: Path, target: str) -> None:
    """Keep old bookmarks, including language, capability and section fragments."""
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="robots" content="noindex"><title>ChemWorld — moved</title>'
        f'<link rel="canonical" href="{html.escape(target, quote=True)}">'
        '<script>const target=new URL('
        + json.dumps(target)
        + ',location.href);target.search=location.search;target.hash=location.hash;'
        'location.replace(target.href);</script>'
        f'<p>This page has moved. <a href="{html.escape(target, quote=True)}">'
        'Continue to ChemWorld</a>.</p></html>\n',
        encoding="utf-8",
    )


def build(output: Path) -> None:
    docs = output / "docs"
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(docs)],
        cwd=ROOT,
        check=True,
    )
    build_explorer(output)
    redirect(output / "explore" / "index.html", "../")
    # Former MkDocs pages were published at the site root.
    count = 0
    for source in docs.rglob("*.html"):
        relative = source.relative_to(docs)
        if relative == Path("index.html"):
            continue
        old = output / relative
        target = os.path.relpath(source, old.parent).replace(os.sep, "/")
        if target.endswith("/index.html"):
            target = target.removesuffix("index.html")
        redirect(old, target)
        count += 1
    (output / ".nojekyll").touch()
    print(f"Website: {output}; documentation: {docs}; legacy redirects: {count + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    build(parser.parse_args().output.resolve())
