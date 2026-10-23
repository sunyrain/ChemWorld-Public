"""Read fixed original deposit blocks through PubChem, without retry or qualification."""

import argparse
import json
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

SELECTION = {
    "3017-68-3": ("DTXSID001312513", 5323715),
    "4192-77-2": ("DTXSID601017688", 637758),
    "2211-68-9": ("DTXSID701037138", 5364761),
    "6776-19-8": ("DTXSID301315530", 5354263),
    "24840-05-9": ("DTXSID601037159", 1711919),
    "31849-78-2": ("DTXSID101350302", 5463114),
    "430-95-5": ("DTXSID10870534", 521133),
    "694-51-9": ("DTXSID70457840", 11182579),
    "15918-07-7": ("DTXSID101376964", 5364664),
}
MATERIAL_SELECTION = (
    ("539-00-4", "HMDB0030325", 92987),
    ("600-07-7", "209-982-7", 8314),
    ("353-42-4", "4888", 2724165),
    ("353-42-4", "dhs_Boron trifluoride compound with methyl ether (1:1)", 2724165),
    ("930-89-2", "DTXSID90873265", 12762847),
    ("2211-67-8", "DTXSID001030848", 75169),
    ("2613-65-2", "DTXSID00879253", 6432281),
    ("2613-66-3", "0002613663", 19502),
    ("6189-41-9", "NSC 24244", 30664),
    ("6553-48-6", "DTXSID201031735", 12175834),
    ("33880-83-0", "f2db41d27d80da0a18f386c111249a05", 10583),
    ("52708-03-9", "DTXSID4074984", 7004865),
    ("625-96-7", "HMDB0033667", 11567),
)


def deposit_fields(record, cas, annotation_name, source_name="EPA DSSTox"):
    """Keep identity-bearing fields; the complete provider payload stays outside Git."""
    database = record.get("source", {}).get("db", {})
    names = record.get("synonyms", [])
    return {
        "SID": record.get("sid"),
        "source_name": database.get("name"),
        "source_ID": database.get("source_id"),
        "source_name_matches": database.get("name") == source_name,
        "source_CAS_in_deposit_names": cas in names,
        "annotation_name_in_deposit_names": annotation_name in names,
        "deposited_names": names,
        "standardized_CIDs": [
            c["id"]["id"]["cid"]
            for c in record.get("compound", [])
            if "cid" in c.get("id", {}).get("id", {})
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubchem", type=Path, required=True)
    parser.add_argument("--consumer-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--block", choices=("nine", "material-scope", "remaining-original"), default="nine"
    )
    parser.add_argument("--previous-directory", type=Path, action="append", default=[])
    args = parser.parse_args()
    pubchem = json.loads(args.pubchem.read_text(encoding="utf8"))
    prior = json.loads(args.consumer_summary.read_text(encoding="utf8"))
    material = args.block == "material-scope"
    remaining = args.block == "remaining-original"
    expected = (
        {"passed": 149, "unresolved": 136} if material else {"passed": 138, "unresolved": 147}
    )
    if remaining:
        expected = {"passed": 155, "unresolved": 130}
    assert prior["total"] == 285 and prior["states"] == expected
    selection = (
        MATERIAL_SELECTION if material else [(cas, *pair) for cas, pair in SELECTION.items()]
    )
    if remaining:
        old = [
            json.loads((directory / "summary.json").read_text(encoding="utf8"))
            for directory in args.previous_directory
        ]
        assert {s["block"] for s in old} == {
            "nine-explicit-DSSTox-deposits",
            "thirteen-material-scope-deposits",
        }
        seen = {
            (q["source_CAS"], q.get("source_name", "EPA DSSTox"), q["source_ID"])
            for s in old
            for q in s["queries"]
        }
        unresolved = {
            r["inputs"]["raw"]["CAS"] for r in prior["results"] if r["status"] == "unresolved"
        }
        assert len(unresolved) == 100
        chosen = {}
        for row in pubchem["original_row_comparison"]:
            cas = row["source_CAS"]
            if cas not in unresolved:
                continue
            for candidate in row["candidates"]:
                for annotation in candidate.get("CAS_matches", []):
                    ref = annotation["reference"]
                    key = cas, ref["SourceName"], ref["SourceID"]
                    if (
                        ref["SourceName"]
                        not in ("EPA DSSTox", "ChemIDplus", "Human Metabolome Database (HMDB)")
                        or key in seen
                    ):
                        continue
                    value = cas, ref["SourceID"], candidate["CID"]
                    assert chosen.setdefault(key, value) == value
        assert len(chosen) == 34 and len({key[0] for key in chosen}) == 25
        assert Counter(key[1] for key in chosen) == {
            "EPA DSSTox": 17,
            "ChemIDplus": 15,
            "Human Metabolome Database (HMDB)": 2,
        }
        selection = list(chosen.values())
    queries = []
    for index, (cas, source_id, cid) in enumerate(selection, 1):
        original = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
        candidate = next(c for c in original["candidates"] if c["CID"] == cid)
        annotation = next(
            a for a in candidate["CAS_matches"] if a["reference"]["SourceID"] == source_id
        )
        assert cas in annotation["CAS"]
        source_name = annotation["reference"]["SourceName"]
        if not material and not remaining:
            assert source_name == "EPA DSSTox"
        units = [
            r["unit"]
            for r in prior["results"]
            if r["inputs"]["raw"]["CAS"] == cas and r["status"] == "unresolved"
        ]
        assert units
        queries.append(
            {
                "source_CAS": cas,
                "source_ID": source_id,
                "source_name": source_name,
                "expected_CID": cid,
                "original_units": units,
                "original_PubChem_candidates": original["candidates"],
                "selected_annotation": annotation,
                "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/"
                f"{quote(source_name, safe='')}/{quote(source_id, safe='')}/JSON",
                "filename": f"{index}-{cas}.json" if material or remaining else f"{source_id}.json",
            }
        )
    count = 13 if material else 9
    if remaining:
        assert len({unit for q in queries for unit in q["original_units"]}) == 26
        for sid, source_name, page in (
            (
                24856830,
                "Sigma-Aldrich",
                "https://www.sigmaaldrich.com/US/en/product/aldrich/278777",
            ),
            (
                87577720,
                "TCI (Tokyo Chemical Industry Co., Ltd.)",
                "https://www.tcichemicals.com/OF/en/p/T2043",
            ),
        ):
            original = next(
                r for r in pubchem["original_row_comparison"] if r["source_CAS"] == "998-41-4"
            )
            queries.append(
                {
                    "source_CAS": "998-41-4",
                    "source_name": source_name,
                    "source_ID": None,
                    "requested_SID": sid,
                    "original_page_url": page,
                    "original_units": [
                        r["unit"]
                        for r in prior["results"]
                        if r["inputs"]["raw"]["CAS"] == "998-41-4"
                    ],
                    "original_PubChem_candidates": original["candidates"],
                    "expected_CID": None,
                    "url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{sid}/JSON",
                    "filename": f"supplier-{sid}.json",
                }
            )
        count = 36
    assert len(queries) == count
    args.output.mkdir(parents=True, exist_ok=False)
    started, denied = time.perf_counter(), 0
    summary = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": (
            "remaining-34-original-deposits-and-two-supplier-SIDs"
            if remaining
            else "thirteen-material-scope-deposits"
            if material
            else "nine-explicit-DSSTox-deposits"
        ),
        "request_total": count,
        "source_rows": len({unit for q in queries for unit in q["original_units"]}),
        "queries": [],
        "new_verified_materials": 0,
    }

    def save():
        summary["seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf8"
        )

    def request(row):
        nonlocal denied
        if denied >= 3:
            row["status"] = "unattempted_after_access_stop"
        else:
            row["retrieved_UTC"] = datetime.now(UTC).isoformat()
            try:
                with urlopen(row["url"], timeout=15) as response:
                    raw = response.read()
                    row["http_status"] = response.status
                    row["content_type"] = response.headers.get("Content-Type")
                (args.output / row["filename"]).write_bytes(raw)
                denied = 0
                row["status"] = "received"
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
        if raw is not None:
            try:
                records = json.loads(raw)["PC_Substances"]
                row["records"] = [
                    deposit_fields(
                        record,
                        row["source_CAS"],
                        row.get("selected_annotation", {})
                        .get("reference", {})
                        .get("Name", "Tributylsilane"),
                        row["source_name"],
                    )
                    for record in records
                ]
            except Exception:
                row.update(status="content_error", error=traceback.format_exc())
        save()
        elapsed = time.perf_counter() - started
        print(
            f"Initial {index}/{count} {row['source_CAS']}: {row['status']} "
            f"{index / elapsed:.2f}/s ETA {elapsed * (count - index) / index:.1f}s",
            flush=True,
        )
        time.sleep(1)
    if material or remaining:
        sids = list(dict.fromkeys(r["SID"]["id"] for q in queries for r in q.get("records", [])))
        summary["sdf_request_total"] = len(sids)
        summary["sdf_queries"] = []
        sdf_started = time.perf_counter()
        for index, sid in enumerate(sids, 1):
            row = {
                "SID": sid,
                "filename": f"{sid}.sdf",
                "url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{sid}/SDF",
            }
            summary["sdf_queries"].append(row)
            raw = request(row)
            if raw is not None:
                row["has_mol_end"] = b"M  END" in raw
                row["record_delimiters"] = raw.count(b"$$$$")
            save()
            elapsed = time.perf_counter() - sdf_started
            print(
                f"SDF {index}/{len(sids)} SID={sid}: {row['status']} "
                f"{index / elapsed:.2f}/s ETA {elapsed * (len(sids) - index) / index:.1f}s",
                flush=True,
            )
            time.sleep(1)
        summary["sdf_states"] = dict(Counter(q["status"] for q in summary["sdf_queries"]))
    summary["states"] = dict(Counter(q["status"] for q in summary["queries"]))
    summary["collection_finished"] = True
    save()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("queries", "sdf_queries")}))


if __name__ == "__main__":
    main()
