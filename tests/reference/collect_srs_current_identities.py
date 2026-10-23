"""Read GSRS identities for every current CAS returned by the fixed SRS block."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

from collect_gsrs_identities import BASE, Collection, public_identity


def collect(srs_path, prior_path, output):
    srs = json.loads(srs_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    assert srs["attempted"] == srs["total"] == 96 and not srs["unattempted_CAS"]
    assert prior["attempted"] == prior["total"] == 234
    references = {}
    for row in srs["results"]:
        for candidate in row.get("candidates", []):
            cas = candidate.get("currentCasNumber")
            if cas:
                references.setdefault(cas, []).append(
                    {
                        "source_CAS": row["source_CAS"],
                        "original_units": row["original_units"],
                        "SRS_ITN": candidate["internalTrackingNumber"],
                        "SRS_CAS_codes": candidate["casNumbers"],
                    }
                )
    cas_numbers = list(references)
    existing = {r["source_CAS"]: r for r in prior["results"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "query-plan.json").write_text(
        json.dumps(
            {
                "current_CAS": references,
                "new_queries": [c for c in cas_numbers if c not in existing],
                "reused_queries": [c for c in cas_numbers if c in existing],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    collection = Collection(output)
    identities, rows = {}, []
    print(
        f"Fixed {len(cas_numbers)} current CAS, "
        f"{sum(c not in existing for c in cas_numbers)} new searches",
        flush=True,
    )
    for cas in cas_numbers:
        row = {
            "current_CAS": cas,
            "original_sources": references[cas],
            "requests": [],
            "candidates": [],
            "status": "collecting",
        }
        if cas in existing:
            old = existing[cas]
            row.update(
                status=old["status"], candidates=old["candidates"], reused_from=str(prior_path)
            )
            for candidate in old["candidates"]:
                uuid = candidate["uuid"]
                identities[uuid] = prior["identities"][uuid]
        else:
            skip = 0
            while True:
                url = (
                    BASE
                    + "/search?"
                    + urlencode(
                        {
                            "q": f'root_codes_code:"^{cas}$"',
                            "top": 100,
                            "skip": skip,
                        }
                    )
                )
                page, request_id = collection.get(url, f"{cas}-search-{skip}.json")
                row["requests"].append(request_id)
                if page is None:
                    row["status"] = "search_request_failed"
                    break
                row["reported_total"] = page["total"]
                row["returned"] = skip + len(page["content"])
                for candidate in page["content"]:
                    uuid = candidate["uuid"]
                    if uuid not in identities:
                        full, request_id = collection.get(candidate["_self"], uuid + ".json")
                        row["requests"].append(request_id)
                        identities[uuid] = (
                            public_identity(full) | {"request": request_id}
                            if full is not None and full.get("uuid") == uuid
                            else {
                                "uuid": uuid,
                                "request": request_id,
                                "error": "full_record_unavailable_or_uuid_mismatch",
                            }
                        )
                    matches = [
                        c
                        for c in identities[uuid].get("codes", []) or []
                        if c.get("codeSystem") == "CAS" and c.get("code") == cas
                    ]
                    row["candidates"].append({"uuid": uuid, "exact_CAS_codes": matches})
                    if collection.denied:
                        break
                if collection.denied:
                    row["status"] = "stopped_after_denied_request"
                    break
                if row["returned"] >= page["total"]:
                    row["status"] = (
                        "candidate_details_failed"
                        if any(identities[c["uuid"]].get("error") for c in row["candidates"])
                        else "exact_code_candidates"
                        if any(c["exact_CAS_codes"] for c in row["candidates"])
                        else "no_exact_CAS_code"
                        if row["candidates"]
                        else "no_candidates"
                    )
                    break
                if not page["content"]:
                    row["status"] = "incomplete_pagination"
                    break
                skip = row["returned"]
        rows.append(row)
        elapsed = time.perf_counter() - collection.started
        report = {
            "task": "NATIVE-CONV-22G1",
            "mode": "development",
            "block": "GSRS-all-SRS-current-CAS",
            "source_artifact": str(srs_path),
            "prior_artifact": str(prior_path),
            "total": len(cas_numbers),
            "completed": len(rows),
            "unattempted_CAS": cas_numbers[len(rows) :],
            "results": rows,
            "identities": identities,
            "requests": collection.requests,
            "states": dict(Counter(r["status"] for r in rows)),
            "seconds": elapsed,
            "stop_reason": "denied_request" if collection.denied else None,
            "qualification": "Identity source collection only; no components approved.",
        }
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            f"Completed {len(rows)}/{len(cas_numbers)} current CAS; "
            f"{len(rows) / elapsed:.3f}/s; "
            f"ETA {elapsed * (len(cas_numbers) - len(rows)) / len(rows):.1f}s",
            flush=True,
        )
        if collection.denied:
            break
    print(
        json.dumps(
            {k: report[k] for k in ("total", "completed", "states", "seconds", "stop_reason")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srs-summary", required=True, type=Path)
    parser.add_argument("--prior-gsrs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.srs_summary, args.prior_gsrs, args.output)
