"""Read fixed public phase-basis and mixture-child identity evidence without retry."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from collect_identity_annotations import annotations

CHILDREN = {
    "96-19-5": (
        "064e67a6-18b4-4e00-9d39-54f82e9877ae",
        "f60e1d6d-0336-4b36-bee2-4cdf9d086d5b",
    ),
    "2396-78-3": (
        "002af4af-8684-4d97-a000-c9524159e4e5",
        "b549f64b-9614-40fa-aed9-215cda695a8d",
    ),
}


def identity_fields(record):
    """Extract identity evidence without copying provider MOL or internal payloads."""
    result = {
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
        )
    }
    result["codes"] = [
        {key: code.get(key) for key in ("codeSystem", "code", "type", "url", "references")}
        for code in record.get("codes", [])
    ]
    result["structure"] = {
        key: record.get("structure", {}).get(key)
        for key in (
            "formula",
            "smiles",
            "_inchi",
            "_inchiKey",
            "mwt",
            "charge",
            "stereochemistry",
            "stereoCenters",
            "definedStereo",
            "ezCenters",
            "opticalActivity",
            "atropisomerism",
        )
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsrs-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf8"))

    gsrs = read(args.gsrs_directory / "summary.json")
    queries = [
        {
            "kind": "sulfur-phase-PDF",
            "url": "https://www.acs.org/content/dam/acsorg/about/governance/committees/chemicalsafety/safetypractices/clip-sulfur.pdf",
            "filename": "acs-sulfur.pdf",
        }
    ]
    sulfur = next(r for r in gsrs["results"] if r["source_CAS"] == "7704-34-9")
    sulfur_record = gsrs["identities"][sulfur["candidates"][0]["uuid"]]
    key = sulfur_record["structure"]["_inchiKey"]
    assert key == "JLQNHALFVCURHW-UHFFFAOYSA-N"
    queries.append(
        {
            "kind": "sulfur-full-key-structure",
            "query_InChIKey": key,
            "GSRS_uuid": sulfur_record["uuid"],
            "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"inchikey/{key}/property/MolecularFormula,MolecularWeight,InChI,InChIKey,IUPACName/JSON",
            "filename": "sulfur-full-key.json",
        }
    )
    for cas, expected in CHILDREN.items():
        result = next(r for r in gsrs["results"] if r["source_CAS"] == cas)
        parent_id = result["candidates"][0]["uuid"]
        parent = read(args.gsrs_directory / f"{parent_id}.json")
        children = parent["mixture"]["components"]
        assert tuple(c["substance"]["refuuid"] for c in children) == expected
        for child in children:
            assert child["type"] == "MUST_BE_PRESENT"
            uid = child["substance"]["refuuid"]
            queries.append(
                {
                    "kind": "GSRS-mixture-child",
                    "parent_CAS": cas,
                    "parent_uuid": parent_id,
                    "child_reference": child,
                    "query_uuid": uid,
                    "url": f"https://gsrs.ncats.nih.gov/api/v1/substances({uid})?view=full",
                    "filename": f"{uid}.json",
                }
            )
    assert len(queries) == 6
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "sulfur-phase-and-four-mixture-children",
        "initial_query_total": 6,
        "queries": [],
        "annotation_queries": [],
        "new_verified_materials": 0,
    }
    started, denied = time.perf_counter(), 0

    def save():
        summary["seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf8"
        )

    def request(row):
        nonlocal denied
        if denied >= 3:
            row["status"] = "unattempted_after_access_stop"
            return None
        row["retrieved_UTC"] = datetime.now(UTC).isoformat()
        try:
            with urlopen(row["url"], timeout=15) as response:
                raw = response.read()
                row["http_status"] = response.status
                row["content_type"] = response.headers.get("Content-Type")
            (args.output / row["filename"]).write_bytes(raw)
            row["status"] = "received"
            denied = 0
            return raw
        except HTTPError as exc:
            (args.output / (row["filename"] + ".error")).write_bytes(exc.read())
            row.update(status="http_error", http_status=exc.code, error=str(exc))
            denied = denied + 1 if exc.code in (401, 403, 429) else 0
        except (URLError, TimeoutError) as exc:
            row.update(status="request_error", error=str(exc))
            denied = 0
        return None

    for index, row in enumerate(queries, 1):
        summary["queries"].append(row)
        raw = request(row)
        if raw is not None and row["kind"] == "sulfur-phase-PDF":
            row["content_review"] = {
                "PDF_header_present": raw.startswith(b"%PDF-"),
                "content_type_is_pdf": row["content_type"].split(";")[0] == "application/pdf",
                "status": "PDF_header_received" if raw.startswith(b"%PDF-") else "not_a_PDF",
                "qualification": "Content type and header are not a scientific review of the PDF.",
            }
        elif raw is not None:
            record = json.loads(raw)
            if row["kind"] == "GSRS-mixture-child":
                row["returned_uuid_matches"] = record.get("uuid") == row["query_uuid"]
                row["record"] = identity_fields(record)
            else:
                row["candidates"] = record.get("PropertyTable", {}).get("Properties", [])
        save()
        elapsed = time.perf_counter() - started
        print(
            f"Initial {index}/6 {row['kind']}: {row['status']} "
            f"{index / elapsed:.2f}/s ETA {elapsed * (6 - index) / index:.1f}s",
            flush=True,
        )
        time.sleep(1)
    cids = list(dict.fromkeys(c["CID"] for c in queries[1].get("candidates", [])))
    summary["annotation_query_total"] = len(cids)
    for index, cid in enumerate(cids, 1):
        row = {
            "CID": cid,
            "filename": f"{cid}-annotations.json",
            "url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=CAS",
        }
        summary["annotation_queries"].append(row)
        raw = request(row)
        if raw is not None:
            record = json.loads(raw)["Record"]
            row["record_CID"] = record["RecordNumber"]
            row["annotations"] = annotations(record)
        save()
        print(f"Annotations {index}/{len(cids)} CID={cid}: {row['status']}", flush=True)
        time.sleep(1)
    summary["states"] = dict(
        Counter(r["status"] for r in summary["queries"] + summary["annotation_queries"])
    )
    summary["collection_finished"] = True
    save()
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("queries", "annotation_queries")})
    )


if __name__ == "__main__":
    main()
