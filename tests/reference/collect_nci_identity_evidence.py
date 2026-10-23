"""Collect public NCI CIR evidence for the fixed remaining GSRS identity gaps."""

import argparse
import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--gsrs", type=Path, required=True)
    parser.add_argument("--pubchem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior, gsrs, pubchem = map(read, (args.prior_summary, args.gsrs, args.pubchem))
    assert prior["states"] == {"passed": 106, "unresolved": 179}
    available = {r["source_CAS"] for r in gsrs["results"] if r["candidates"]}
    gaps = [
        r
        for r in prior["results"]
        if r["status"] == "unresolved" and r["inputs"]["raw"]["CAS"] not in available
    ]
    numbers = list(dict.fromkeys(r["inputs"]["raw"]["CAS"] for r in gaps))
    assert len(numbers) == 131 and len(gaps) == 170
    args.output.mkdir(parents=True, exist_ok=False)
    started, blocked = time.perf_counter(), 0
    requests = []
    result = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "NCI-CIR-remaining-identities",
        "CAS_total": 131,
        "source_rows": 170,
        "request_total": 262,
        "requests": requests,
        "results": [],
        "complete_and_passed": False,
        "qualification": "Candidate evidence only; no components approved automatically.",
    }
    for cas in numbers:
        entry = {"source_CAS": cas, "representations": {}}
        for representation in ("stdinchi", "names"):
            url = f"https://cactus.nci.nih.gov/chemical/structure/{cas}/{representation}/xml"
            row = {"source_CAS": cas, "representation": representation, "url": url}
            entry["representations"][representation] = row
            requests.append(row)
            if blocked >= 3:
                row["status"] = "unattempted_after_access_stop"
                continue
            row["retrieved_UTC"] = datetime.now(UTC).isoformat()
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    data = response.read()
                    row["http_status"] = response.status
                (args.output / f"{cas}-{representation}.xml").write_bytes(data)
                blocked = 0
                root = ET.fromstring(data)
                assert root.tag == "request" and root.attrib["string"] == cas
                assert root.attrib["representation"] == representation
                row["candidates"] = [
                    {**node.attrib, "items": [item.text for item in node.findall("item")]}
                    for node in root.findall("data")
                ]
                row["status"] = "received"
            except urllib.error.HTTPError as exc:
                row.update(status="http_error", http_status=exc.code, error=str(exc))
                (args.output / f"{cas}-{representation}.error").write_bytes(exc.read())
                blocked = blocked + 1 if exc.code in (401, 403, 429) else 0
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                blocked = 0
            elapsed = time.perf_counter() - started
            rate = len(requests) / max(elapsed, 1e-9)
            print(
                f"{len(requests)}/262 {cas} {representation}: {row['status']} "
                f"{rate:.2f} requests/s ETA {(262 - len(requests)) / rate:.0f}s",
                flush=True,
            )
            result.update(
                request_completed=len(requests),
                seconds=elapsed,
                request_states=dict(Counter(r["status"] for r in requests)),
            )
            (args.output / "summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf8",
            )
            time.sleep(1)
        entry["original_units"] = [r["unit"] for r in gaps if r["inputs"]["raw"]["CAS"] == cas]
        candidates = next(
            r["candidates"] for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas
        )
        values = [
            item
            for group in entry["representations"]["stdinchi"].get("candidates", [])
            for item in group["items"]
        ]
        entry["PubChem_candidates"] = [
            {
                "CID": c["CID"],
                "InChI": c["InChI"],
                "InChIKey": c["InChIKey"],
                "NCI_complete_InChI_match": c["InChI"] in values,
            }
            for c in candidates
        ]
        result["results"].append(entry)
    states = dict(Counter(r["status"] for r in requests))
    result.update(
        request_completed=sum(r["status"] != "unattempted_after_access_stop" for r in requests),
        request_states=states,
        seconds=time.perf_counter() - started,
        complete_and_passed=states.get("received", 0) == 262,
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8",
    )
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("requests", "results")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
