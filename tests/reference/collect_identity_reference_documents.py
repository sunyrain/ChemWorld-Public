"""Capture three fixed institutional documents; search every unresolved CAS."""

import argparse
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

DOCUMENTS = (
    (
        "ineris-vanadium.pdf",
        "https://substances.ineris.fr/sites/default/files/archives/"
        "1314-62-1%20--%20pentaoxyde%20de%20divanadium%20--%20FDTE.pdf",
    ),
    (
        "nite-615-38-3.html",
        "https://www.nite.go.jp/chem/jcheck/detail.action?"
        "cno=615-38-3&mno=3-2301&request_locale=en",
    ),
    ("nist-enzyme-review.pdf", "https://srd.nist.gov/jpcrdreprint/1.2789450.pdf"),
)


class TextPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def collect(output, prior):
    assert prior["states"] == {"passed": 158, "unresolved": 127} and prior["total"] == 285
    cases = {}
    for row in prior["results"]:
        if row["status"] == "unresolved":
            cases.setdefault(row["inputs"]["raw"]["CAS"], []).append(row["unit"])
    assert len(cases) == 97
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.perf_counter()
    for index, (filename, url) in enumerate(DOCUMENTS, 1):
        print(f"Document {index}/3: requesting {filename}", flush=True)
        row = {"filename": filename, "url": url, "retrieved_UTC": datetime.now(UTC).isoformat()}
        try:
            with urlopen(url, timeout=20) as response:
                raw = response.read()
                row.update(
                    http_status=response.status, content_type=response.headers.get_content_type()
                )
            (output / filename).write_bytes(raw)
            if filename.endswith(".pdf"):
                if not raw.startswith(b"%PDF-"):
                    raise ValueError("Expected PDF signature, received a different document")
                text_path = output / (filename + ".txt")
                process = subprocess.run(
                    [
                        "pdftotext",
                        "-layout",
                        "-enc",
                        "UTF-8",
                        str(output / filename),
                        str(text_path),
                    ],
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                row["extraction_returncode"] = process.returncode
                row["extraction_stderr"] = process.stderr.decode("utf-8", errors="replace")
                if process.returncode:
                    raise ValueError("pdftotext extraction failed")
                pages = text_path.read_text(encoding="utf-8").split("\f")
                if not pages[-1].strip():
                    pages.pop()
            else:
                parser = TextPage()
                parser.feed(raw.decode("utf-8"))
                pages = ["\n".join(parser.parts)]
                (output / (filename + ".txt")).write_text(pages[0], encoding="utf-8")
            row.update(status="received", bytes=len(raw), pages=len(pages), CAS_matches={})
            for cas in cases:
                matches = []
                for number, page in enumerate(pages, 1):
                    normalized = (
                        page.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-")
                    )
                    if re.search(r"(?<!\d)" + re.escape(cas) + r"(?!\d)", normalized):
                        matches.append(number)
                row["CAS_matches"][cas] = matches
        except HTTPError as exc:
            row.update(status="http_error", http_status=exc.code, error=str(exc))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        elapsed = time.perf_counter() - started
        print(
            f"Completed {index}/3, status={row['status']}, {index / elapsed:.3f} docs/s, "
            f"ETA {elapsed * (3 - index) / index:.1f}s",
            flush=True,
        )
        if index < len(DOCUMENTS):
            time.sleep(1)
    report = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "three-institutional-documents-all-97-unresolved-CAS",
        "expected_requests": 3,
        "attempted_requests": len(rows),
        "source_rows": 127,
        "original_units_by_CAS": cases,
        "results": rows,
        "seconds": time.perf_counter() - started,
        "qualification": "Document capture and text matches only. No components qualified.",
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                r["filename"]: {
                    cas: pages for cas, pages in r.get("CAS_matches", {}).items() if pages
                }
                for r in rows
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.output, json.loads(args.consumer_summary.read_text(encoding="utf-8")))
