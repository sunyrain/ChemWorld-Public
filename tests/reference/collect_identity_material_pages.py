"""Read fixed original identity pages, preserving every unsuccessful response."""

import argparse
import json
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


class TextPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lines = []

    def handle_data(self, text):
        if text.strip():
            self.lines.append(text.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("material", "original-structures"), default="material")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    queries = [
        {
            "source_CAS": "539-00-4",
            "filename": "HMDB0030325.xml",
            "url": "https://hmdb.ca/metabolites/HMDB0030325.xml",
        },
        {
            "source_CAS": "625-96-7",
            "filename": "HMDB0033667.xml",
            "url": "https://hmdb.ca/metabolites/HMDB0033667.xml",
        },
        {
            "source_CAS": "353-42-4",
            "filename": "cameo-4888.html",
            "url": "https://cameochemicals.noaa.gov/chemical/4888",
        },
    ]
    original_structures = args.block == "original-structures"
    if original_structures:
        queries = [
            {
                "source_CAS": "998-41-4",
                "filename": "sigma-278777.html",
                "url": "https://www.sigmaaldrich.com/US/en/product/aldrich/278777",
            },
            {
                "source_CAS": "998-41-4",
                "filename": "tci-T2043.html",
                "url": "https://www.tcichemicals.com/OF/en/p/T2043",
            },
            {
                "source_CAS": "17082-12-1",
                "filename": "spectrabase-7OYvJ2Xr1dw.html",
                "url": "https://spectrabase.com/spectrum/7OYvJ2Xr1dw",
            },
        ]
    started, denied = time.perf_counter(), 0
    result = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "two-original-structures-and-formula-page"
        if original_structures
        else "two-HMDB-XML-and-CAMEO-page",
        "total": 3,
        "results": [],
    }
    for index, row in enumerate(queries, 1):
        result["results"].append(row)
        row["retrieved_UTC"] = datetime.now(UTC).isoformat()
        try:
            if denied >= 3:
                row["status"] = "unattempted_after_access_stop"
                continue
            with urlopen(row["url"], timeout=15) as response:
                raw = response.read()
                row.update(
                    http_status=response.status, content_type=response.headers.get("Content-Type")
                )
            (args.output / row["filename"]).write_bytes(raw)
            denied = 0
            if row["filename"].endswith(".xml"):
                document = ET.fromstring(raw)
                row["document_root"] = document.tag
                wanted = {
                    "accession",
                    "name",
                    "cas_registry_number",
                    "chemical_formula",
                    "smiles",
                    "inchi",
                    "inchikey",
                }
                row["identity_fields"] = {
                    element.tag.rsplit("}", 1)[-1]: element.text
                    for element in document
                    if element.tag.rsplit("}", 1)[-1] in wanted
                }
            else:
                page = TextPage()
                page.feed(raw.decode("utf8"))
                (args.output / Path(row["filename"]).with_suffix(".txt")).write_text(
                    "\n".join(page.lines), encoding="utf8"
                )
                if original_structures:
                    row["text_saved"] = True
                else:
                    selected = set()
                    for i, line in enumerate(page.lines):
                        if any(
                            word.lower() in line.lower()
                            for word in ("353-42-4", "formula", "etherate")
                        ):
                            selected.update(range(max(0, i - 1), min(len(page.lines), i + 4)))
                    row["identity_context"] = [page.lines[i] for i in sorted(selected)]
            row["status"] = "received"
        except HTTPError as exc:
            (args.output / (row["filename"] + ".error")).write_bytes(exc.read())
            row.update(status="http_error", http_status=exc.code, error=str(exc))
            denied = denied + 1 if exc.code in (401, 403, 429) else 0
        except (URLError, TimeoutError) as exc:
            row.update(status="request_error", error=str(exc))
            denied = 0
        except Exception:
            row.update(status="content_error", error=traceback.format_exc())
        result["seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf8"
        )
        print(f"{index}/3 {row['filename']}: {row['status']}", flush=True)
        time.sleep(1)
    result["collection_finished"] = True
    result["seconds"] = time.perf_counter() - started
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
