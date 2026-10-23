"""Collect fixed full-key or NCI-InChI source blocks, without retry or qualification."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from collect_identity_annotations import annotations

SELECTION = {
    "1501-60-6": "RCJMVGJKROQDCB-PLNGDYQASA-N",
    "2613-66-3": "PQXAPVOKLYINEI-SFYZADRCSA-N",
    "6236-88-0": "CYISMTMRBPPERU-KYZUINATSA-N",
    "22104-78-5": "AYQPVPFZWIQERS-UHFFFAOYSA-N",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sander-summary", type=Path)
    parser.add_argument("--nci-summary", type=Path)
    parser.add_argument("--srs-summary", type=Path)
    parser.add_argument("--pubchem", type=Path)
    parser.add_argument("--consumer-summary", type=Path)
    parser.add_argument("--original-structures", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf8"))

    queries, reused_sources, existing_annotations = [], [], {}
    properties = "MolecularFormula,MolecularWeight,InChI,InChIKey,IUPACName"
    if args.srs_summary:
        source, prior, pubchem = map(read, (
            args.srs_summary, args.consumer_summary, args.pubchem,
        ))
        assert prior["states"] == {"passed": 219, "unresolved": 66}
        unresolved = {r["inputs"]["raw"]["CAS"] for r in prior["results"]
                      if r["status"] == "unresolved"}
        old = {r["source_CAS"]: r for r in pubchem["original_row_comparison"]}
        existing_annotations = {r["CID"]: r for r in pubchem["results"]}
        for row in source["results"]:
            cas = row["source_CAS"]
            if cas not in unresolved or not row.get("candidates"):
                continue
            assert len(row["candidates"]) == 1
            current = row["candidates"][0]["currentCasNumber"]
            if current == cas:
                reused_sources.append(old[cas])
                continue
            queries.append({
                "source_CAS": cas, "query_CAS": current,
                "original_units": row["original_units"],
                "SRS_ITN": row["candidates"][0]["internalTrackingNumber"],
                "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"name/{current}/property/{properties}/JSON?name_type=complete",
            })
        assert len(queries) == 5 and len(reused_sources) == 12
        block = "SRS-remaining-five-current-CAS-structures"
    elif args.original_structures:
        assert args.sander_summary is None and args.nci_summary is None
        prior = read(args.consumer_summary)
        assert prior["total"] == 285 and prior["states"] == {"passed": 155, "unresolved": 130}
        for cas, key in (
            ("998-41-4", "HIAZFYQNGXRLTF-UHFFFAOYSA-N"),
            ("17082-12-1", "DMLAVOWQYNRWNQ-BUHFOSPRSA-N"),
        ):
            units = [
                r["unit"]
                for r in prior["results"]
                if r["inputs"]["raw"]["CAS"] == cas and r["status"] == "unresolved"
            ]
            assert len(units) == 1
            queries.append(
                {
                    "source_CAS": cas,
                    "query_InChIKey": key,
                    "original_units": units,
                    "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                    f"inchikey/{key}/property/{properties}/JSON",
                }
            )
        block = "two-original-material-full-key-structures"
    elif args.nci_summary:
        assert args.sander_summary is None
        nci, pubchem, prior = map(read, (args.nci_summary, args.pubchem, args.consumer_summary))
        assert prior["total"] == 285 and prior["states"] == {"passed": 123, "unresolved": 162}
        remaining = {
            r["inputs"]["raw"]["CAS"] for r in prior["results"] if r["status"] == "unresolved"
        }
        for source in nci["results"]:
            cas = source["source_CAS"]
            if cas not in remaining:
                continue
            old = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
            for candidate in source["representations"]["stdinchi"]["candidates"]:
                for inchi in candidate["items"]:
                    if any(inchi == c["InChI"] for c in old["candidates"]):
                        continue
                    queries.append(
                        {
                            "source_CAS": cas,
                            "query_InChI": inchi,
                            "NCI_candidate": candidate,
                            "original_units": source["original_units"],
                            "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                            f"inchi/property/{properties}/JSON?" + urlencode({"inchi": inchi}),
                        }
                    )
        assert len(queries) == len({r["source_CAS"] for r in queries}) == 35
        assert sum(len(r["original_units"]) for r in queries) == 50
        block = "remaining-35-NCI-InChI-structures"
    else:
        source = read(args.sander_summary)
        saved = {r["source_CAS"]: r for r in source["results"]}
        for cas, key in SELECTION.items():
            assert saved[cas]["status"] == "complete_identity_fields"
            assert saved[cas]["identity_fields"]["CASRN"] == cas
            assert saved[cas]["identity_fields"]["InChIKey"] == key
            queries.append(
                {
                    "source_CAS": cas,
                    "query_InChIKey": key,
                    "original_units": saved[cas]["original_units"],
                    "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                    f"inchikey/{key}/property/{properties}/JSON",
                }
            )
        block = "four-Sander-complete-key-structures"
    args.output.mkdir(parents=True, exist_ok=False)
    started, denied = time.perf_counter(), 0
    summary = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": block,
        "structure_query_total": len(queries),
        "structure_queries": [],
        "annotation_queries": [],
        "new_verified_materials": 0,
    }
    if args.srs_summary:
        summary["reused_original_sources"] = reused_sources
        summary["original_annotation_artifact"] = str(args.pubchem)
    denial_limit = 1 if args.srs_summary else 3

    def save():
        summary["seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf8"
        )

    def request(row, filename):
        nonlocal denied
        if denied >= denial_limit:
            row["status"] = "unattempted_after_access_stop"
            return None
        row["retrieved_UTC"] = datetime.now(UTC).isoformat()
        try:
            with urlopen(row["url"], timeout=15) as response:
                raw = response.read()
                row["http_status"] = response.status
            (args.output / filename).write_bytes(raw)
            denied = 0
            result = json.loads(raw)
            row["status"] = "received"
            return result
        except HTTPError as exc:
            (args.output / (filename + ".error")).write_bytes(exc.read())
            row.update(status="http_error", http_status=exc.code, error=str(exc))
            denied = denied + 1 if exc.code in (401, 403, 429) else 0
        except (URLError, TimeoutError, ValueError) as exc:
            denied = 0
            row.update(status="request_error", error=str(exc))
        return None

    for index, row in enumerate(queries, 1):
        cas = row["source_CAS"]
        summary["structure_queries"].append(row)
        response = request(row, f"{cas}-properties.json")
        if response is not None:
            row["candidates"] = response.get("PropertyTable", {}).get("Properties", [])
            if "query_CAS" not in row:
                field = "InChI" if "query_InChI" in row else "InChIKey"
                row["all_graphs_match" if field == "InChI" else "all_keys_match"] = bool(
                    row["candidates"]
                ) and all(c.get(field) == row[f"query_{field}"] for c in row["candidates"])
        save()
        elapsed = time.perf_counter() - started
        print(
            f"Structures {index}/{len(queries)} {cas}: {row['status']} "
            f"{index / elapsed:.2f}/s ETA {elapsed * (len(queries) - index) / index:.1f}s",
            flush=True,
        )
        time.sleep(1)
    cids = list(
        dict.fromkeys(
            c["CID"] for r in summary["structure_queries"] for c in r.get("candidates", [])
        )
    )
    summary["annotation_query_total"] = len(cids)
    annotation_started = time.perf_counter()
    for index, cid in enumerate(cids, 1):
        row = {
            "CID": cid,
            "url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=CAS",
        }
        summary["annotation_queries"].append(row)
        if cid in existing_annotations:
            row.update(existing_annotations[cid], reused_from=str(args.pubchem))
            response = None
        else:
            response = request(row, f"{cid}-annotations.json")
        if response is not None:
            record = response.get("Record", {})
            row["record_CID"] = record.get("RecordNumber")
            row["annotations"] = annotations(record)
            row["record_matches"] = row["record_CID"] == cid
        save()
        elapsed = time.perf_counter() - annotation_started
        print(
            f"Annotations {index}/{len(cids)} CID={cid}: {row['status']} "
            f"{index / elapsed:.2f}/s ETA {elapsed * (len(cids) - index) / index:.1f}s",
            flush=True,
        )
        time.sleep(1)
    for kind in ("structure", "annotation"):
        rows = summary[f"{kind}_queries"]
        summary[f"{kind}_states"] = dict(Counter(r["status"] for r in rows))
        summary[f"{kind}_attempted"] = sum(
            r["status"] != "unattempted_after_access_stop" for r in rows
        )
    summary["collection_finished"] = True
    save()
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_queries")}))


if __name__ == "__main__":
    main()
