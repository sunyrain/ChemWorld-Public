"""Collect fixed remaining identity fields, without importing Henry-law values."""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class IdentityPage(HTMLParser):
    """Retain inline chemical names and formula subscripts in their source cells."""

    def __init__(self):
        super().__init__()
        self.heading = []
        self.headings = []
        self.in_heading = False
        self.row_stack = []
        self.cell_stack = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "h1":
            self.heading = []
            self.in_heading = True
        elif tag == "tr":
            self.row_stack.append([])
        elif tag in ("td", "th"):
            self.cell_stack.append([])

    def handle_data(self, data):
        if self.in_heading:
            self.heading.append(data)
        if self.cell_stack:
            self.cell_stack[-1].append(data)

    def handle_endtag(self, tag):
        if tag == "h1":
            self.in_heading = False
            self.headings.append(" ".join("".join(self.heading).split()))
        elif tag in ("td", "th") and self.cell_stack:
            value = " ".join("".join(self.cell_stack.pop()).split())
            if self.row_stack:
                self.row_stack[-1].append(value)
        elif tag == "tr" and self.row_stack:
            self.rows.append(self.row_stack.pop())

    def identity(self):
        fields = {}
        for row in self.rows:
            if len(row) != 2:
                continue
            label = row[0].replace(" ", "").rstrip(":")
            if label in ("FORMULA", "CASRN", "InChIKey"):
                fields[label] = row[1]
            elif label.startswith("STRUCTURE"):
                fields["structure_attribution"] = row[0]
        title = next((heading for heading in self.headings if "\u2192" in heading), "")
        return {"page_heading": title, "name": title.rsplit("\u2192", 1)[-1].strip(), **fields}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--pubchem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior, pubchem = map(read, (args.prior_summary, args.pubchem))
    assert prior["total"] == 285 and prior["states"] == {"passed": 118, "unresolved": 167}
    gaps = [r for r in prior["results"] if r["status"] == "unresolved"]
    numbers = list(dict.fromkeys(r["inputs"]["raw"]["CAS"] for r in gaps))
    assert len(numbers) == 128 and len(gaps) == 167
    args.output.mkdir(parents=True, exist_ok=False)
    started, denied = time.perf_counter(), 0
    results = []
    summary = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "Sander-remaining-128-CAS",
        "request_total": 128,
        "source_rows": 167,
        "qualification": (
            "Identity fields only; no materials approved and no Henry values imported."
        ),
        "results": results,
    }
    for cas in numbers:
        url = f"https://henrys-law.org/henry/casrn/{cas}"
        row = {"source_CAS": cas, "url": url}
        results.append(row)
        row["original_units"] = [r["unit"] for r in gaps if r["inputs"]["raw"]["CAS"] == cas]
        if denied >= 3:
            row["status"] = "unattempted_after_access_stop"
            continue
        row["retrieved_UTC"] = datetime.now(UTC).isoformat()
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                raw = response.read()
                row["http_status"] = response.status
            denied = 0
            (args.output / f"{cas}.html").write_bytes(raw)
            page = IdentityPage()
            page.feed(raw.decode("utf8"))
            fields = page.identity()
            row["identity_fields"] = fields
            cas_matches = cas in re.findall(r"\b\d{2,7}-\d{2}-\d\b", fields.get("CASRN", ""))
            key = fields.get("InChIKey", "")
            row["status"] = (
                "complete_identity_fields"
                if cas_matches
                and fields.get("FORMULA")
                and re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key)
                else "incomplete_or_different_identity_fields"
            )
            candidates = next(
                r["candidates"]
                for r in pubchem["original_row_comparison"]
                if r["source_CAS"] == cas
            )
            row["PubChem_candidates"] = [
                {
                    "CID": c["CID"],
                    "InChIKey": c["InChIKey"],
                    "full_key_match": key == c["InChIKey"],
                }
                for c in candidates
            ]
        except urllib.error.HTTPError as exc:
            row.update(status="http_error", http_status=exc.code, error=str(exc))
            (args.output / f"{cas}.error").write_bytes(exc.read())
            denied = denied + 1 if exc.code in (401, 403, 429) else 0
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            denied = 0
        elapsed = time.perf_counter() - started
        rate = len(results) / max(elapsed, 1e-9)
        print(
            f"{len(results)}/128 {cas}: {row['status']} "
            f"{rate:.2f} requests/s ETA {(128 - len(results)) / rate:.0f}s",
            flush=True,
        )
        summary.update(
            request_completed=len(results),
            states=dict(Counter(r["status"] for r in results)),
            seconds=elapsed,
        )
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
        )
        time.sleep(1)
    summary.update(
        request_completed=sum(r["status"] != "unattempted_after_access_stop" for r in results),
        states=dict(Counter(r["status"] for r in results)),
        seconds=time.perf_counter() - started,
        collection_finished=True,
        new_verified_materials=0,
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
