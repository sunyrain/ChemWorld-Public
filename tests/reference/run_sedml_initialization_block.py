"""Fixed six full native experiments, existing SDK reports and isolated 60-digit forms."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

IDENTITIES = tuple(kind+"-reset-"+str(reset).lower()
                   for kind in ("I", "E", "IE") for reset in (True, False))
SEDML = "{http://sed-ml.org/sed-ml/level1/version4}"
CORE = "{http://www.sbml.org/sbml/level3/version2/core}"


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def closed(sources, output):
    import mpmath as mp

    mp.mp.dps = 60
    report = {"version": mp.__version__, "digits": 60, "units": []}
    for identity in IDENTITIES:
        kind, reset_text = identity.split("-reset-")
        reset = reset_text == "true"
        experiment = ET.fromstring((sources/(identity+".sedml")).read_text(encoding="utf-8"))
        source = ET.fromstring((sources/(kind+".xml")).read_text(encoding="utf-8"))
        params = {p.get("id"): mp.mpf(p.get("value"))
                  for p in source.findall(CORE+"model/"+CORE+"listOfParameters/"+CORE+"parameter")}
        scan = experiment.find(SEDML+"listOfTasks/"+SEDML+"repeatedTask")
        assert scan.get("resetModel") == reset_text
        vector = scan.find(SEDML+"listOfRanges/"+SEDML+"vectorRange")
        values = [mp.mpf(v.text) for v in vector.findall(SEDML+"value")]
        simulation = experiment.find(SEDML+"listOfSimulations/"+SEDML+"uniformTimeCourse")
        assert (simulation.get("initialTime"), simulation.get("outputStartTime"),
                simulation.get("outputEndTime"), simulation.get("numberOfSteps")) == (
                    "0", "0", "1", "4")
        assert values == list(map(mp.mpf, ("0", ".5", "3")))
        assert params["k"] == 2 and params["rho"] == mp.mpf(".1")
        species = source.find(CORE+"model/"+CORE+"listOfSpecies/"+CORE+"species")
        default_a = mp.mpf(species.get("initialConcentration"))
        event = source.find(CORE+"model/"+CORE+"listOfEvents/"+CORE+"event")
        delay = mp.mpf(".25")
        if event is not None:
            trigger = event.find(CORE+"trigger")
            assert trigger.get("initialValue") == "false" and trigger.get("persistent") == "true"
        assert (event is not None) == ("E" in kind)
        assert bool(source.findall(CORE+"model/"+CORE+"listOfInitialAssignments")) == ("I" in kind)
        jumps = []
        if "E" in kind and not reset:
            jumps.append((delay, params["k"]))
            previous = params["k"] > 1
            for index, value in enumerate(values):
                truth = value > 1
                if truth and not previous:
                    jumps.append((mp.mpf(index)+delay, value))
                previous = truth
        rows = []
        for index, k in enumerate(values):
            for sample in range(5):
                local_time = mp.mpf(sample)/4
                t = local_time if reset else mp.mpf(index)+local_time
                initial_a = 2*(k if reset else params["k"]) if "I" in kind else default_a
                active_jumps = ([(delay, k)] if "E" in kind and k > 1 else []) if reset else jumps
                eligible = [(when, value) for when, value in active_jumps if when <= t]
                if eligible:
                    when, value = eligible[-1]
                    a = value*mp.exp(-params["rho"]*(t-when))
                else:
                    a = initial_a*mp.exp(-params["rho"]*t)
                rows.append([mp.nstr(value, 65) for value in (t, a, k)])
        report["units"].append({"id": identity, "rows_decimal": rows})
        write(output/"closed.json", report)
        print(f"closed initialization {len(report['units'])}/6", flush=True)


def native(sources, output):
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.data.sedml_declarations import SEDMLDocument
    from chemworld.eval.sedml_experiment import run_sedml_experiment
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.world.sedml_protocol import SEDMLProtocol

    provenance = InputSource(DataSourceProvenance("sedml-six-initialization", 0,
        "Original self-authored fixed six SED-ML initialization experiments", "1", "CC0-1.0"),
        "synthetic")
    report = {"expected_experiments": 6, "expected_rows": 90, "expected_values": 270,
              "expected_native_operations": 171, "units": []}
    started = time.perf_counter()
    for identity in IDENTITIES:
        kind = identity.split("-reset-")[0]
        unit = {"id": identity, "rows": [], "failures": [], "history": [], "complete": False}
        report["units"].append(unit)
        try:
            document = SEDMLDocument((sources/(identity+".sedml")).read_text(encoding="utf-8"),
                                     provenance)
            model = SBMLModel((sources/(kind+".xml")).read_text(encoding="utf-8"), provenance)
            plan = SEDMLProtocol(document, {kind+".xml": model})
            unit["compiled"] = True
            result = run_sedml_experiment(plan, seed=37)
            unit["history"] = [asdict(record) for record in result.history]
            unit["reports"], unit["iterations"] = result.reports, result.iterations
            unit["snapshots"] = result.snapshots
            columns = {column["label"]: column["data"] for column in result.reports["report"]}
            if set(columns) != {"t", "a", "kin"} or any(
                    value["shape"] != [15] for value in columns.values()):
                raise ValueError("complete native experiment report has unexpected columns or axes")
            rows = list(map(list, zip(*(columns[key]["values"] for key in ("t", "a", "kin")),
                                     strict=True)))
            if len(rows) != 15 or not all(math.isfinite(v) for row in rows for v in row):
                raise ValueError("native experiment must produce all fifteen finite report rows")
            unit["rows"] = rows
            unit["complete"] = True
        except Exception as exc:
            unit["failures"].append(traceback.format_exc())
            if hasattr(exc, "history"):
                unit["history"] = [asdict(record) for record in exc.history]
            for key in ("snapshots", "snapshot", "iterations", "variable_arrays"):
                if hasattr(exc, key):
                    unit[key] = getattr(exc, key)
        write(output/"native.json", report)
        print(f"native initialization {len(report['units'])}/6 rows={len(unit['rows'])}/15 "
              f"complete={unit['complete']} elapsed={time.perf_counter()-started:.2f}s", flush=True)
    report["row_count"] = sum(len(unit["rows"]) for unit in report["units"])
    report["value_count"] = 3*report["row_count"]
    report["operation_count"] = sum(len(unit["history"]) for unit in report["units"])
    report["complete_experiments"] = sum(unit["complete"] for unit in report["units"])
    report["runtime_contract_passed"] = (report["complete_experiments"] == 6
        and report["value_count"] == 270 and report["operation_count"] == 171
        and all(record["info"]["operation_outcome"] == "success"
                for unit in report["units"] for record in unit["history"]))
    write(output/"native.json", report)
    return report


def compare(actual, expected, *, rtol, atol):
    units = []
    for identity in IDENTITIES:
        left = next((u for u in actual["units"] if u["id"] == identity), {}).get("rows", [])
        target = next((u for u in expected["units"] if u["id"] == identity), {})
        right = target.get("rows", target.get("rows_decimal", []))
        unit = {"id": identity, "executed": 0, "unexecuted": 0, "comparisons": []}
        for row in range(15):
            for column in range(3):
                if row >= len(left) or row >= len(right):
                    unit["unexecuted"] += 1
                    continue
                observed, reference = float(left[row][column]), float(right[row][column])
                error = abs(observed-reference)
                limit = atol+rtol*abs(reference)
                unit["executed"] += 1
                unit["comparisons"].append({"row": row, "column": column, "observed": observed,
                    "expected": reference, "absolute_error": error, "limit": limit,
                    "passed": math.isfinite(error) and error <= limit})
        unit["passed"] = unit["executed"] == 45 and all(c["passed"] for c in unit["comparisons"])
        units.append(unit)
    return {"expected_experiments": 6, "expected_comparisons": 270,
        "executed_comparisons": sum(unit["executed"] for unit in units),
        "unexecuted_comparisons": sum(unit["unexecuted"] for unit in units),
        "passed_experiments": sum(unit["passed"] for unit in units),
        "passed_comparisons": sum(c["passed"] for unit in units for c in unit["comparisons"]),
        "complete_and_passed": all(unit["passed"] for unit in units), "units": units}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copasi", type=Path)
    parser.add_argument("--tellurium", type=Path)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--stage", choices=("parent", "closed"), default="parent")
    args = parser.parse_args()
    if args.stage == "closed":
        closed(args.source_directory, args.output)
        return 0
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root) or any(
            value is None for value in (args.copasi, args.tellurium, args.reference_python)):
        raise ValueError("outside-Git evidence, both SDK reports and isolated mpmath are required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output/"inputs"
    shutil.copytree(args.source_directory, sources)
    hashes = {}
    for path in sorted(sources.iterdir()):
        original = path.read_bytes()
        if any((ref/"inputs"/path.name).read_bytes() != original
               for ref in (args.copasi, args.tellurium)):
            raise ValueError("native and both SDK references must use identical complete inputs")
        hashes[path.name] = hashlib.sha256(original).hexdigest()
    write(args.output/"source_bindings.json", hashes)
    shutil.copy2(__file__, args.output/"executed_source.py")
    note = root/"workstreams/world_foundation/NATIVE_CONV50C_EXPERIMENT.md"
    shutil.copy2(note, args.output/note.name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    write(args.output/"binding.json", {"source_commit": commit,
        "copasi": str(args.copasi), "tellurium": str(args.tellurium), "mode": "development"})
    copasi = json.loads((args.copasi/"reference.json").read_text(encoding="utf-8"))
    tellurium = json.loads((args.tellurium/"reference.json").read_text(encoding="utf-8"))
    if not all(data.get("collection_complete") for data in (copasi, tellurium)):
        raise ValueError("both unchanged SDK references must retain their full six-experiment data")
    native_report = native(sources, args.output)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [str(args.reference_python), "-I", str(Path(__file__).resolve()), "--stage", "closed",
               "--source-directory", str(sources), "--output", str(args.output)]
    with ((args.output/"closed.log").open("x", encoding="utf-8") as log,
          subprocess.Popen(command, stdout=log, stderr=log, env=environment) as child):
        while True:
            try:
                child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f"closed reference alive pid={child.pid} log_bytes={log.tell()}", flush=True)
    write(args.output/"closed_process.json", {"exit": child.returncode, "command": command})
    closed_path = args.output/"closed.json"
    forms = (json.loads(closed_path.read_text(encoding="utf-8")) if closed_path.exists()
             else {"units": []})
    families = {"native_to_closed": compare(native_report, forms, rtol=2e-7, atol=2e-9),
        "COPASI_to_closed": compare(copasi, forms, rtol=5e-6, atol=5e-8),
        "Tellurium_to_closed": compare(tellurium, forms, rtol=5e-6, atol=5e-8),
        "native_to_COPASI": compare(native_report, copasi, rtol=5e-6, atol=5e-8),
        "native_to_Tellurium": compare(native_report, tellurium, rtol=5e-6, atol=5e-8)}
    write(args.output/"comparisons.json", families)
    summary = {"expected_experiments": 6, "expected_rows": 90, "expected_values": 270,
        "expected_native_operations": 171, "expected_all_comparisons": 1350,
        "closed_reference_exit": child.returncode,
        **{key: native_report[key] for key in ("row_count", "value_count", "operation_count",
                                              "complete_experiments", "runtime_contract_passed")},
        "comparisons": {key: {k: v for k, v in value.items() if k != "units"}
                        for key, value in families.items()},
        "Tellurium_algorithm_parity_passed": False,
        "complete_and_passed": native_report["runtime_contract_passed"]
                               and child.returncode == 0
                               and all(value["complete_and_passed"] for value in families.values())}
    write(args.output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
