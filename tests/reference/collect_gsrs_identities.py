"""Collect public GSRS records for the fixed original F2 identity gaps."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

BASE = "https://gsrs.ncats.nih.gov/api/v1/substances"


def public_identity(record):
    """Keep identity and cited evidence, not a guessed CAS-to-structure mapping."""
    structure = record.get("structure")
    return {
        key: record.get(key)
        for key in (
            "uuid",
            "approvalID",
            "substanceClass",
            "definitionType",
            "definitionLevel",
            "status",
            "version",
            "deprecated",
            "_name",
            "names",
            "codes",
            "references",
        )
    } | {
        "structure": {k: v for k, v in structure.items() if k != "molfile"} if structure else None,
    }


class Collection:
    def __init__(self, output):
        self.output = output
        self.started = time.perf_counter()
        self.requests = []
        self.denied = 0

    def get(self, url, filename):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "gsrs.ncats.nih.gov"
            or not (
                parsed.path == "/api/v1/substances/search"
                or parsed.path.startswith("/api/v1/substances(")
            )
        ):
            raise ValueError("unexpected GSRS public record URL")
        if self.requests:
            time.sleep(1)
        request = {"url": url, "retrieved_UTC": datetime.now(UTC).isoformat()}
        data = None
        try:
            with urlopen(url, timeout=15) as response:
                raw = response.read()
                request["http_status"] = response.status
            (self.output / filename).write_bytes(raw)
            data = json.loads(raw)
            request["status"] = "received"
        except HTTPError as exc:
            (self.output / (filename + ".http-error.txt")).write_bytes(exc.read())
            request.update(status="http_error", http_status=exc.code, error=str(exc))
        except (URLError, TimeoutError, ValueError) as exc:
            request.update(status="request_error", error=str(exc))
        self.denied = self.denied + 1 if request.get("http_status") in {401, 403, 429} else 0
        self.requests.append(request)
        print(
            f"GSRS request {len(self.requests)} {filename}: {request['status']} "
            f"HTTP={request.get('http_status')} elapsed={time.perf_counter() - self.started:.1f}s",
            flush=True,
        )
        return data, len(self.requests) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf8"))
    cas_numbers = list(dict.fromkeys(r["source_CAS"] for r in inventory["results"]))
    assert inventory["total"] == 285 and len(cas_numbers) == 234
    args.output.mkdir(parents=True, exist_ok=False)
    collection = Collection(args.output)
    rows, identities = [], {}
    for cas in cas_numbers:
        row = {"source_CAS": cas, "requests": [], "candidates": [], "status": "collecting"}
        skip, returned = 0, 0
        while True:
            url = (
                BASE
                + "/search?"
                + urlencode(
                    {
                        "q": f'root_codes_code:"^{cas}$"',
                        "top": 100,
                        "skip": skip,
                        "includeFacets": "false",
                    }
                )
            )
            page, request_id = collection.get(url, f"{cas}-search-{skip}.json")
            row["requests"].append(request_id)
            if page is None:
                row["status"] = "search_request_failed"
                break
            row["reported_total"] = page["total"]
            candidates = page["content"]
            returned += len(candidates)
            row["returned"] = returned
            for candidate in candidates:
                uuid = candidate["uuid"]
                if uuid not in identities:
                    full, request_id = collection.get(candidate["_self"], f"{uuid}.json")
                    row["requests"].append(request_id)
                    if full is not None and full.get("uuid") == uuid:
                        identities[uuid] = public_identity(full) | {"request": request_id}
                    else:
                        identities[uuid] = {
                            "uuid": uuid,
                            "request": request_id,
                            "error": "full_record_unavailable_or_uuid_mismatch",
                        }
                full = identities[uuid]
                matches = [
                    c
                    for c in full.get("codes", []) or []
                    if c.get("codeSystem") == "CAS" and c.get("code") == cas
                ]
                row["candidates"].append({"uuid": uuid, "exact_CAS_codes": matches})
                if collection.denied >= 3:
                    break
            if collection.denied >= 3:
                row["status"] = "stopped_after_denied_requests"
                break
            if returned >= page["total"]:
                row["status"] = (
                    "exact_code_candidates"
                    if any(c["exact_CAS_codes"] for c in row["candidates"])
                    else "candidate_details_failed"
                    if any(identities[c["uuid"]].get("error") for c in row["candidates"])
                    else "no_exact_CAS_code"
                    if row["candidates"]
                    else "no_candidates"
                )
                break
            if not candidates:
                row["status"] = "incomplete_pagination"
                break
            skip += len(candidates)
        rows.append(row)
        elapsed = time.perf_counter() - collection.started
        result = {
            "task": "NATIVE-CONV-22G1",
            "mode": "development",
            "block": "GSRS-234-CAS",
            "total": 234,
            "attempted": len(rows),
            "unattempted_CAS": cas_numbers[len(rows) :],
            "results": rows,
            "identities": identities,
            "requests": collection.requests,
            "states": dict(Counter(r["status"] for r in rows)),
            "seconds": elapsed,
            "new_verified_mappings": 0,
            "complete_and_verified": False,
            "stop_reason": "three_consecutive_denied_requests"
            if collection.denied >= 3
            else "completed"
            if len(rows) == 234
            else "running",
        }
        (args.output / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf8",
        )
        print(
            f"{len(rows)}/234 CAS; {len(rows) / elapsed:.2f}/s "
            f"ETA {elapsed * (234 - len(rows)) / len(rows):.1f}s",
            flush=True,
        )
        if collection.denied >= 3:
            break
    print(
        json.dumps(
            {k: result[k] for k in ("total", "attempted", "states", "seconds", "stop_reason")}
        )
    )


if __name__ == "__main__":
    main()
