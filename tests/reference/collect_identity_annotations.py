"""Trace fixed external CAS candidates to PubChem's original annotation sources."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def annotations(record):
    references = {r["ReferenceNumber"]: r for r in record.get("Reference", [])}
    result = []

    def visit(sections):
        for section in sections:
            if section.get("TOCHeading") == "CAS":
                for info in section.get("Information", []):
                    reference = references.get(info.get("ReferenceNumber"), {})
                    result.append(
                        {
                            "CAS": [
                                v["String"]
                                for v in info.get("Value", {}).get("StringWithMarkup", [])
                            ],
                            "annotation_url": info.get("URL"),
                            "reference": {
                                k: reference.get(k)
                                for k in (
                                    "ReferenceNumber",
                                    "SourceName",
                                    "SourceID",
                                    "Name",
                                    "URL",
                                    "LicenseURL",
                                )
                            },
                        }
                    )
            visit(section.get("Section", []))

    visit(record.get("Section", []))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    properties = json.loads(args.properties.read_text(encoding="utf8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf8"))
    assert properties["total"] == properties["attempted"] == 234
    assert properties["stop_reason"] == "completed" and inventory["total"] == 285
    cids = list(
        dict.fromkeys(c["CID"] for r in properties["results"] for c in r.get("candidates", []))
    )
    assert len(cids) == 204
    args.output.mkdir(parents=True, exist_ok=False)
    rows, consecutive_denied = [], 0
    start = time.perf_counter()
    for i, cid in enumerate(cids, 1):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=CAS"
        row = {"CID": cid, "url": url, "retrieved_UTC": datetime.now(UTC).isoformat()}
        try:
            with urlopen(url, timeout=15) as response:
                raw = response.read()
                row["http_status"] = response.status
            (args.output / f"{cid}.json").write_bytes(raw)
            record = json.loads(raw)["Record"]
            if record["RecordNumber"] != cid:
                raise ValueError("returned CID differs from requested CID")
            row["annotations"] = annotations(record)
            row["status"] = "annotations_received" if row["annotations"] else "no_annotations"
        except HTTPError as exc:
            (args.output / f"{cid}.http-error.txt").write_bytes(exc.read())
            row.update(status="http_error", http_status=exc.code, error=str(exc))
        except (URLError, TimeoutError, ValueError, KeyError) as exc:
            row.update(status="request_error", error=str(exc))
        rows.append(row)
        consecutive_denied = (
            consecutive_denied + 1 if row.get("http_status") in {401, 403, 429} else 0
        )
        result = {
            "task": "NATIVE-CONV-22G1",
            "mode": "development",
            "block": "CAS-annotations",
            "total": 204,
            "attempted": len(rows),
            "unattempted_CIDs": cids[len(rows) :],
            "results": rows,
            "new_verified_mappings": 0,
            "complete_and_verified": False,
            "stop_reason": "three_consecutive_denied_requests"
            if consecutive_denied >= 3
            else "completed"
            if len(rows) == 204
            else "running",
            "seconds": time.perf_counter() - start,
        }
        (args.output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        elapsed = time.perf_counter() - start
        print(
            f"{i}/204 CID={cid} {row['status']} HTTP={row.get('http_status')} "
            f"{i / elapsed:.2f}/s ETA {elapsed * (204 - i) / i:.1f}s",
            flush=True,
        )
        if consecutive_denied >= 3:
            break
        if i < len(cids):
            time.sleep(1)
    by_cid = {r["CID"]: r for r in rows}
    by_cas = {r["source_CAS"]: r for r in properties["results"]}
    comparisons = []
    for original in inventory["results"]:
        cas = original["source_CAS"]
        candidates = []
        for candidate in by_cas[cas].get("candidates", []):
            key = candidate["InChIKey"]
            annotations_row = by_cid.get(candidate["CID"], {})
            candidates.append(
                {
                    **candidate,
                    "annotation_status": annotations_row.get("status", "not_attempted"),
                    "CAS_matches": [
                        a for a in annotations_row.get("annotations", []) if cas in a["CAS"]
                    ],
                    "local_exact_key_matches": [
                        c["record_id"]
                        for c in original.get("candidates", [])
                        if c["inchi_key"] == key
                    ],
                    "local_other_keys": list(
                        dict.fromkeys(
                            c["inchi_key"]
                            for c in original.get("candidates", [])
                            if c["inchi_key"] != key
                        )
                    ),
                }
            )
        comparisons.append(
            {
                "unit": original["unit"],
                "source_CAS": cas,
                "source_name": original["source_name"],
                "candidates": candidates,
                "status": "no_external_candidates"
                if not candidates
                else "source_CAS_annotation_present"
                if any(c["CAS_matches"] for c in candidates)
                else "source_CAS_not_annotated",
                "known_source_conflict": "NIST InChI stereo /m conflicts with CAS and PubChem"
                if cas == "54-11-5"
                else None,
            }
        )
    result["original_row_comparison"] = comparisons
    result["row_states"] = dict(Counter(r["status"] for r in comparisons))
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"results", "original_row_comparison", "unattempted_CIDs"}
            }
        )
    )


if __name__ == "__main__":
    main()
