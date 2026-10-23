"""Capture all remaining original CAS queries through EPA's documented SRS API."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

SERVER = "https://cdxapps.epa.gov/oms-substance-registry-services"
SCHEMA_URL = (
    SERVER + "/oms-substance-registry-services/v3/api-docs/SRS%20REST"
)
PARAMETERS = "?searchAllCas=true&includeOtherCas=true&includeRelationships=true"
FIELDS = (
    "internalTrackingNumber", "systematicName", "epaIdentificationNumber",
    "currentCasNumber", "epaName", "substanceType", "categoryClass", "iupacName",
    "pubChemId", "dtxsid", "comptoxUpdateDate", "molecularWeight", "molecularFormula",
    "inchiNotation", "smilesNotation", "classifications", "characteristics",
    "synonyms", "casNumbers", "relationships",
)


def collect(prior, schema_path, output):
    assert prior["total"] == 285 and prior["states"] == {"passed": 159, "unresolved": 126}
    cases = {}
    for row in prior["results"]:
        if row["status"] == "unresolved":
            cases.setdefault(row["inputs"]["raw"]["CAS"], []).append(row["unit"])
    assert len(cases) == 96
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes)
    assert schema["servers"][0]["url"] == SERVER
    api = schema["paths"]["/rest-api/substance/cas/{casNum}"]["get"]
    assert {"searchAllCas", "includeOtherCas", "includeRelationships"} <= {
        parameter["name"] for parameter in api["parameters"]
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "api-schema.json").write_bytes(schema_bytes)
    started = time.perf_counter()
    rows = []
    stop_reason = None
    for index, (cas, units) in enumerate(cases.items(), 1):
        url = SERVER + "/rest-api/substance/cas/" + cas + PARAMETERS
        row = {
            "source_CAS": cas, "original_units": units, "url": url,
            "retrieved_UTC": datetime.now(UTC).isoformat(), "filename": cas + ".json",
        }
        print(f"Request {index}/96: {cas}", flush=True)
        try:
            with urlopen(url, timeout=20) as response:
                raw = response.read()
                row.update(
                    http_status=response.status,
                    content_type=response.headers.get_content_type(),
                )
            (output / row["filename"]).write_bytes(raw)
            data = json.loads(raw)
            assert isinstance(data, list), "Expected API array of substance records"
            assert all(isinstance(record, dict) for record in data)
            row["candidates"] = [
                {key: record[key] for key in FIELDS if key in record} for record in data
            ]
            row["status"] = "received_candidates" if data else "no_candidates"
            row["candidate_count"] = len(data)
        except HTTPError as exc:
            raw = exc.read()
            (output / (cas + ".error")).write_bytes(raw)
            row.update(status="http_error", http_status=exc.code, error=str(exc))
            if exc.code in (401, 403, 429):
                stop_reason = f"HTTP{exc.code}; stop without retry or alternate client"
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        elapsed = time.perf_counter() - started
        report = {
            "task": "NATIVE-CONV-22G1", "mode": "development",
            "block": "EPA-SRS-all-96-remaining-CAS",
            "schema_url": SCHEMA_URL, "schema_version": schema["info"]["version"],
            "total": 96, "source_rows": 126, "attempted": len(rows),
            "unattempted_CAS": list(cases)[len(rows):], "results": rows,
            "states": dict(Counter(r["status"] for r in rows)),
            "seconds": elapsed, "stop_reason": stop_reason,
            "qualification": "Source capture only; no component identity approved.",
        }
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            f"Completed {index}/96: {row['status']}; {index / elapsed:.3f} CAS/s; "
            f"ETA {elapsed * (96 - index) / index:.1f}s",
            flush=True,
        )
        if stop_reason:
            break
        if index < 96:
            time.sleep(1)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-summary", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(json.loads(args.consumer_summary.read_text(encoding="utf-8")), args.schema, args.output)
