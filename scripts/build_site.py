"""Build the original integrated website at /, with legacy document redirects."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

from build_explorer_site import GUIDES
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


def guide_redirect(page: Path, root: Path, guide: str, language: str) -> None:
    """Send previously published document URLs back to the integrated reader."""
    target = os.path.relpath(root, page.parent).replace(os.sep, "/") + "/"
    route = f"guides/chemworld/{guide}"
    canonical = f"{target}?lang={language}#{route}"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="robots" content="noindex"><title>ChemWorld documentation</title>'
        f'<link rel="canonical" href="{html.escape(canonical, quote=True)}">'
        '<script>const target=new URL('
        + json.dumps(target)
        + ',location.href);target.search=location.search;'
        'const language=target.searchParams.get("lang");'
        'if(!["en","zh"].includes(language))target.searchParams.set("lang",'
        + json.dumps(language)
        + ');let anchor=location.hash.slice(1);'
        'try{anchor=decodeURIComponent(anchor);}catch{}target.hash='
        + json.dumps(route)
        + '+(anchor?"/"+encodeURIComponent(anchor):"");location.replace(target.href);</script>'
        f'<p><a href="{html.escape(canonical, quote=True)}">'
        'Open ChemWorld documentation</a></p></html>\n',
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
    # Main guides use the original reader; other technical pages retain a static fallback.
    count = 0
    pages = list(docs.rglob("*.html"))
    for source in pages:
        relative = source.relative_to(docs)
        parts = relative.parts
        language = "zh" if parts[0] == "zh" else "en"
        local = parts[1:] if language == "zh" else parts
        guide = "quickstart" if local == ("index.html",) else local[0]
        old = output / relative
        if guide in GUIDES and len(local) <= 2:
            guide_redirect(source, output, guide, language)
            count += 1
            if relative != Path("index.html"):
                guide_redirect(old, output, guide, language)
                count += 1
            continue
        target = os.path.relpath(source, old.parent).replace(os.sep, "/")
        if target.endswith("/index.html"):
            target = target.removesuffix("index.html")
        redirect(old, target)
        count += 1
    (output / ".nojekyll").touch()
    print(f"Integrated website: {output}; legacy redirects: {count + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    build(parser.parse_args().output.resolve())
