"""Read fixed F2 CAS identities from public sources; never assign components."""

import argparse
import html
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def nist_candidates(raw):
    text = raw.decode("utf8")
    cas = re.search(r"CAS Registry Number:</strong>\s*([^<]*)", text)
    records = []
    for block in re.findall(
        r'<script\s+type="application/ld\+json"[^>]*>(.*?)</script>', text, re.S | re.I
    ):
        entity = json.loads(block)
        for value in entity if isinstance(entity, list) else [entity]:
            if value.get("@type") != "MolecularEntity":
                continue
            records.append(
                {
                    "record_CAS": html.unescape(cas.group(1)).strip() if cas else None,
                    "source_name": value.get("name"),
                    "MolecularFormula": value.get("molecularFormula"),
                    "MolecularWeight_with_unit": value.get("molecularWeight"),
                    "InChI": value.get("inChI"),
                    "InChIKey": value.get("inChIKey"),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("pubchem", "nist"), default="pubchem")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf8"))
    cas_numbers = list(dict.fromkeys(r["source_CAS"] for r in inventory["results"]))
    assert inventory["total"] == 285 and len(cas_numbers) == 234
    args.output.mkdir(parents=True, exist_ok=False)
    rows, consecutive_denied = [], 0
    started = time.perf_counter()
    for i, cas in enumerate(cas_numbers, 1):
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/"
            "MolecularFormula,MolecularWeight,InChI,InChIKey,IUPACName/JSON?name_type=complete"
        )
        if args.source == "nist":
            url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas.replace('-', '')}&Units=SI"
        row = {"source_CAS": cas, "url": url, "retrieved_UTC": datetime.now(UTC).isoformat()}
        try:
            with urlopen(url, timeout=15) as response:
                raw = response.read()
                row["http_status"] = response.status
            suffix = "html" if args.source == "nist" else "json"
            (args.output / f"{cas}.{suffix}").write_bytes(raw)
            row["candidates"] = (
                nist_candidates(raw)
                if args.source == "nist"
                else json.loads(raw).get("PropertyTable", {}).get("Properties", [])
            )
            row["status"] = "unverified_candidates" if row["candidates"] else "no_candidates"
        except HTTPError as exc:
            (args.output / f"{cas}.http-error.txt").write_bytes(exc.read())
            row.update(status="http_error", http_status=exc.code, error=str(exc))
        except (URLError, TimeoutError, ValueError) as exc:
            row.update(status="request_error", error=str(exc))
        rows.append(row)
        denied = row.get("http_status") in {401, 403, 429}
        consecutive_denied = consecutive_denied + 1 if denied else 0
        result = {
            "task": "NATIVE-CONV-22G1",
            "mode": "development",
            "block": f"{args.source}-234-CAS",
            "total": 234,
            "attempted": len(rows),
            "unattempted_CAS": cas_numbers[len(rows) :],
            "results": rows,
            "complete_and_verified": False,
            "new_verified_mappings": 0,
            "stop_reason": "three_consecutive_denied_requests"
            if consecutive_denied >= 3
            else "completed"
            if len(rows) == 234
            else "running",
            "seconds": time.perf_counter() - started,
        }
        (args.output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        elapsed = time.perf_counter() - started
        print(
            f"{i}/234 {cas}: {row['status']} HTTP={row.get('http_status')} "
            f"{i / elapsed:.2f}/s ETA {elapsed * (234 - i) / i:.1f}s",
            flush=True,
        )
        if consecutive_denied >= 3:
            break
        if i != len(cas_numbers):
            time.sleep(1)
    print(json.dumps({k: v for k, v in result.items() if k not in {"results", "unattempted_CAS"}}))


if __name__ == "__main__":
    main()
