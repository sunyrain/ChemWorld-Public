"""Encode explicitly sourced racemates with the official reference InChI CLI."""

import argparse
import json
import re
import subprocess
import time
import traceback
from pathlib import Path

CAS_NUMBERS = (
    "1755-01-7",
    "491-07-6",
    "932-44-5",
    "933-48-2",
    "2402-06-4",
    "6284-84-0",
    "6982-25-8",
    "7667-60-9",
)


def stereochemical_layers(inchi):
    return [part for part in inchi.split("/")[1:] if not part.startswith(("m", "s"))]


def convert_depositors(args):
    """Survey every returned original deposit without asserting a material definition."""
    source = json.loads((args.depositor_directory / "summary.json").read_text(encoding="utf8"))
    remaining = source["block"] == "remaining-34-original-deposits-and-two-supplier-SIDs"
    assert source["collection_finished"]
    if remaining:
        assert source["request_total"] == len(source["queries"]) == 36
        modes = ("absolute",)
    else:
        assert source["block"] == "thirteen-material-scope-deposits"
        assert source["states"] == {"received": 8, "http_error": 5}
        assert len(source["sdf_queries"]) == 8 and source["sdf_states"] == {"received": 8}
        modes = ("absolute", "racemic")
    total = len(source["sdf_queries"]) * len(modes)
    args.output.mkdir(parents=True, exist_ok=False)
    started, rows = time.perf_counter(), []
    for sdf in source["sdf_queries"]:
        sid = sdf["SID"]
        query = next(
            q for q in source["queries"] if any(r["SID"]["id"] == sid for r in q.get("records", []))
        )
        candidate = next(
            (c for c in query["original_PubChem_candidates"] if c["CID"] == query["expected_CID"]),
            None,
        )
        assert candidate is not None or (remaining and query.get("requested_SID") == sid)
        input_file = args.depositor_directory / sdf["filename"]
        for mode in modes:
            prefix = args.output / f"{sid}-{mode}"
            output, log, problem = [str(prefix) + suffix for suffix in (".txt", ".log", ".prb")]
            options = ["/Key", "/AuxNone", "/NoLabels"] + (["/SRac"] if mode == "racemic" else [])
            command = [str(args.inchi), str(input_file), output, log, problem, *options]
            row = {
                "source_CAS": query["source_CAS"],
                "source_name": query["source_name"],
                "source_ID": query["source_ID"],
                "SID": sid,
                "mode": mode,
                "options": options,
                "source_formula": candidate["MolecularFormula"] if candidate else None,
                "expected_CID": candidate["CID"] if candidate else None,
                "source_InChI": candidate["InChI"] if candidate else None,
                "source_InChIKey": candidate["InChIKey"] if candidate else None,
                "source_url": sdf["url"],
                "command": command,
            }
            try:
                assert sdf["status"] == "received" and sdf["has_mol_end"]
                run = subprocess.run(command, capture_output=True, text=True, timeout=15)
                Path(str(prefix) + ".stdout").write_text(run.stdout, encoding="utf8")
                Path(str(prefix) + ".stderr").write_text(run.stderr, encoding="utf8")
                row.update(returncode=run.returncode, program_report=run.stderr)
                assert run.returncode == 0
                output_text = Path(output).read_text(encoding="utf8")
                (inchi,) = re.findall(r"^InChI=.*$", output_text, re.M)
                (key,) = re.findall(r"^InChIKey=([A-Z-]+)$", output_text, re.M)
                inchi = inchi.strip()
                assert re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key)
                row.update(
                    inchi=inchi,
                    inchi_key=key,
                    complete_graph_matches_expected_CID=(
                        inchi == candidate["InChI"] and key == candidate["InChIKey"]
                    )
                    if candidate
                    else None,
                    has_tetrahedral_layer="/t" in inchi,
                    racemic_layer="/s3" in inchi,
                    has_absolute_enantiomer_layer=any(
                        s.startswith("m") for s in inchi.split("/")[1:]
                    ),
                    graph_except_absolute_layers_matches=(
                        stereochemical_layers(inchi) == stereochemical_layers(candidate["InChI"])
                    )
                    if candidate
                    else None,
                    status="encoded",
                )
                if remaining:
                    row.update(
                        formula_from_InChI=inchi.split("/")[1],
                        has_double_bond_layer="/b" in inchi,
                        original_candidate_matches=[
                            c["CID"]
                            for c in query["original_PubChem_candidates"]
                            if c["InChI"] == inchi and c["InChIKey"] == key
                        ],
                    )
            except Exception:
                row.update(status="failed", failure=traceback.format_exc())
            rows.append(row)
            print(f"{len(rows)}/{total} SID={sid} {mode}: {row['status']}", flush=True)
    result = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "remaining-depositor-default-SDF-encodings"
        if remaining
        else "eight-depositor-SDF-encodings",
        "reference_version": "InChI 1.07.5",
        "reference_url": "https://github.com/IUPAC-InChI/InChI/releases/tag/v1.07.5",
        "source_artifact": str(args.depositor_directory / "summary.json"),
        "total": total,
        "completed": len(rows),
        "encoded": sum(r["status"] == "encoded" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "results": rows,
        "seconds": time.perf_counter() - started,
        "identity_qualification": (
            "Encoding is not qualification; retain every graph mismatch and undefined scope."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    return bool(result["failed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsrs-directory", type=Path)
    parser.add_argument("--pubchem", type=Path)
    parser.add_argument("--depositor-directory", type=Path)
    parser.add_argument("--srs-fenchone", action="store_true")
    parser.add_argument("--inchi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.depositor_directory:
        assert args.gsrs_directory is None and args.pubchem is None and not args.srs_fenchone
        return convert_depositors(args)
    assert args.gsrs_directory is not None and args.pubchem is not None
    gsrs = json.loads((args.gsrs_directory / "summary.json").read_text(encoding="utf8"))
    pubchem = json.loads(args.pubchem.read_text(encoding="utf8"))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.perf_counter()
    cas_numbers = ("18492-37-0",) if args.srs_fenchone else CAS_NUMBERS
    total = 2 * len(cas_numbers)
    for cas in cas_numbers:
        if args.srs_fenchone:
            candidate = next(r for r in gsrs["results"] if r["current_CAS"] == "1195-79-5")
            assert candidate["status"] == "exact_code_candidates"
            assert any(s["source_CAS"] == cas for s in candidate["original_sources"])
        else:
            candidate = next(r for r in gsrs["results"] if r["source_CAS"] == cas)
        assert len(candidate["candidates"]) == 1
        uuid = candidate["candidates"][0]["uuid"]
        raw = json.loads((args.gsrs_directory / (uuid + ".json")).read_text(encoding="utf8"))
        structure = raw["structure"]
        assert raw["substanceClass"] == "chemical" and raw["definitionLevel"] == "COMPLETE"
        assert structure["stereochemistry"] == "RACEMIC"
        if args.srs_fenchone:
            assert uuid == "2266125a-3987-4cda-b319-dabefe16d6fc"
            assert raw["status"] == "approved" and not raw["deprecated"]
            assert structure["stereoCenters"] == structure["definedStereo"] == 2
            assert structure["opticalActivity"] == "( + / - )"
            assert any(c["codeSystem"] == "CAS" and c["code"] == "1195-79-5"
                       and c["type"] == "PRIMARY" for c in raw["codes"])
        input_file = args.output / (cas + ".mol")
        input_file.write_text(structure["molfile"], encoding="utf8")
        key = "structure_queries" if args.srs_fenchone else "original_row_comparison"
        candidates = next(r for r in pubchem[key] if r["source_CAS"] == cas)
        if args.srs_fenchone:
            assert candidates["query_CAS"] == "1195-79-5" and candidates["status"] == "received"
        for mode in ("absolute", "racemic"):
            prefix = args.output / (cas + "-" + mode)
            output, log, problem = [str(prefix) + suffix for suffix in (".txt", ".log", ".prb")]
            options = ["/Key", "/AuxNone", "/NoLabels"] + (["/SRac"] if mode == "racemic" else [])
            command = [str(args.inchi), str(input_file), output, log, problem, *options]
            row = {
                "source_CAS": cas,
                "GSRS_uuid": uuid,
                "GSRS_version": raw["version"],
                "source_name": raw["_name"],
                "mode": mode,
                "options": options,
                "source_InChI": structure["_inchi"],
                "source_InChIKey": structure["_inchiKey"],
                "source_formula": structure["formula"],
                "command": command,
            }
            try:
                run = subprocess.run(command, capture_output=True, text=True, timeout=15)
                Path(str(prefix) + ".stdout").write_text(run.stdout, encoding="utf8")
                Path(str(prefix) + ".stderr").write_text(run.stderr, encoding="utf8")
                row["returncode"] = run.returncode
                row["program_report"] = run.stderr
                assert run.returncode == 0
                text = Path(output).read_text(encoding="utf8")
                inchi = re.search(r"^InChI=.*$", text, re.M).group().strip()
                key = re.search(r"^InChIKey=([A-Z-]+)$", text, re.M).group(1)
                row.update(inchi=inchi, inchi_key=key)
                assert re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key)
                if args.srs_fenchone:
                    row["current_PubChem_comparisons"] = [{
                        "CID": c["CID"], "InChI": c["InChI"], "InChIKey": c["InChIKey"],
                        "complete_graph_matches": c["InChI"] == inchi and c["InChIKey"] == key,
                        "relative_layers_match": (
                            stereochemical_layers(c["InChI"]) == stereochemical_layers(inchi)
                        ),
                    } for c in candidates["candidates"]]
                if mode == "absolute":
                    assert inchi == structure["_inchi"] and key == structure["_inchiKey"]
                    if not args.srs_fenchone:
                        assert any(
                            c["InChI"] == inchi and c["InChIKey"] == key
                            for c in candidates["candidates"]
                        )
                else:
                    assert inchi.startswith("InChI=1/") and "/s3" in inchi
                    assert not any(part.startswith("m") for part in inchi.split("/")[1:])
                    assert stereochemical_layers(inchi) == stereochemical_layers(
                        structure["_inchi"]
                    )
                    assert key != structure["_inchiKey"]
                row["status"] = "passed"
            except Exception:
                row.update(status="failed", failure=traceback.format_exc())
            rows.append(row)
            print(f"{len(rows)}/{total} {cas} {mode}: {row['status']}", flush=True)
    result = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "SRS-fenchone-InChI-encoding" if args.srs_fenchone else "racemate-InChI-encoding",
        "reference_version": "InChI 1.07.5",
        "reference_url": "https://github.com/IUPAC-InChI/InChI/releases/tag/v1.07.5",
        "total": total,
        "completed": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "results": rows,
        "seconds": time.perf_counter() - started,
        "identity_qualification": (
            "Encoding checks do not resolve source naming conflicts or qualify consumers."
        ),
    }
    if args.srs_fenchone:
        result["GSRS_source_directory"] = str(args.gsrs_directory)
        result["current_PubChem_source"] = str(args.pubchem)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
