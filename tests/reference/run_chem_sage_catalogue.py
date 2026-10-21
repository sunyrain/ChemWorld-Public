"""Complete original ChemSage catalogue: valid numerical cases and source refusals.

Source coverage, applicable numerical qualification and original data validity
have separate outcomes. A refused phase is never counted as six numerical passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

FILES = ("2026-Dixon-Na-K-Cl-I.dat", "Kaye_Pd-Ru-Tc-Mo.dat", "KF-NIF2_switched.dat",
         "MQMQA-tern-tests.dat", "Ocadiz-Flores.dat", "Shishin_Fe-Sb-O-S_slag.dat", "Viitala.dat")
FAMILIES = {"PURE": 75, "QKTO": 14, "IDMX": 1, "SUBL": 3, "SUBG": 16, "SUBQ": 5}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True), encoding="utf-8")


def reference_catalogue(directory, output):
    import pycalphad
    from pycalphad.io.cs_dat import parse_cs_dat

    spec = importlib.util.spec_from_file_location("independent_source_checks",
        Path(__file__).with_name("mqmqa_material_reference.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if pycalphad.__version__ != "0.11.2":
        raise ValueError("catalogue reference must be pycalphad 0.11.2")
    report = {"reference_version": pycalphad.__version__, "files": [], "phases": []}
    for name in FILES:
        path = directory/name
        header, solutions, pure = parse_cs_dat(path.read_text(encoding="utf-8-sig").upper())
        originals = (*solutions, *pure)
        counts, seen = Counter(p.phase_name for p in originals), Counter()
        report["files"].append({"name": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "phases": len(originals)})
        for index, raw in enumerate(originals):
            seen[raw.phase_name] += 1
            phase = (raw.phase_name if counts[raw.phase_name] == 1 else
                     f"{raw.phase_name}#OCCURRENCE{seen[raw.phase_name]}")
            family = raw.phase_type or "PURE"
            issues = helper.source_issues(raw, header) if family in {"SUBG", "SUBQ"} else []
            limits = []
            for member in raw.endmembers:
                previous = .01
                for interval in member.intervals:
                    upper = 10000. if previous == interval.T_max else interval.T_max
                    previous = interval.T_max
                limits.append(upper)
            high = min(2000., *limits)
            if high <= 300:
                raise ValueError("original phase has no required common temperature interval")
            report["phases"].append({"database": name, "phase": phase,
                "source_label": raw.phase_name, "source_phase_index": index, "family": family,
                "expected": "source_refusal" if issues else "numerical",
                "source_issues": issues, "scheduled_points": 6,
                "temperatures": [300.+(high-300.)*f for f in (.2, .5, .8)]})
        write(output, report)
        print(f"reference catalogue {len(report['files'])}/7 sources "
              f"{len(report['phases'])}/114 phases", flush=True)


def quantity_unit(label):
    """Report units without adding a second property/formula evaluator."""
    quantity, *axes = label.split("|")
    if quantity in {"H", "response:G"} or quantity.startswith("contribution:"):
        unit = "J/mol_nonvacant_atoms"
    elif quantity in {"S", "Cp_fixed"}:
        unit = "J/(mol_nonvacant_atoms*K)"
    elif quantity.startswith("ion_n:"):
        unit = "mol_ions/mol_quadruplets"
    else:
        unit = "dimensionless"
    for axis in axes:
        if axis == "temperature":
            unit += "/K"
        elif axis == "pressure":
            unit += "/Pa"
    return unit


def assess(catalogue, structure, cef, mq, planned_hashes):
    result = {"expected_files": 7, "expected_phases": 114, "expected_original_points": 684,
              "expected_valid_phases": 111, "expected_valid_points": 666,
              "expected_source_refusals": 3, "units": [], "failures": []}
    errors = result["failures"]
    if catalogue.get("reference_version") != "0.11.2":
        errors.append("independent reference version is not the fixed 0.11.2")
    cases = catalogue.get("phases", [])
    identities = [(c["database"], c["phase"]) for c in cases]
    if len(cases) != 114 or len(set(identities)) != len(identities):
        errors.append("reference catalogue must contain exactly 114 unique original phases")
    if dict(Counter(c["family"] for c in cases)) != FAMILIES:
        errors.append("reference catalogue family counts differ from the fixed full source")
    if any(c.get("scheduled_points") != 6 or len(set(c.get("temperatures", []))) != 3
           for c in cases):
        errors.append("original per-phase temperature/profile coverage changed")
    files = catalogue.get("files", [])
    if len(files) != 7 or {f["name"] for f in files} != set(FILES):
        errors.append("reference catalogue must contain all seven original files once")
    for file in files:
        if file.get("sha256") != planned_hashes.get(file["name"]):
            errors.append("original source bytes changed: "+file["name"])
        if file["phases"] != sum(c["database"] == file["name"] for c in cases):
            errors.append("source phase count mismatch: "+file["name"])
    inputs = structure.get("files", [])
    if (len(inputs) != 7 or {f["name"] for f in inputs} != set(FILES)
            or not all(f.get("passed") and not f.get("failures") and not f.get("differences")
                       for f in inputs)):
        errors.append("complete original input structure did not pass for all seven files")
    if (sum(f.get("phases", 0) for f in inputs) != 114
            or sum(f.get("endmembers", 0) for f in inputs) != 202
            or sum(f.get("parameters", 0) for f in inputs) != 170):
        errors.append("structural phase/endmember/excess coverage changed")
    for f in inputs:
        if f.get("sha256") != planned_hashes.get(f["name"]):
            errors.append("structural input bytes differ: "+f["name"])
    native_units = {}
    for block, report, expected in (("cef", cef, 93), ("mqmqa", mq, 21)):
        rows = report.get("units", [])
        keys = [(u["database"], u["phase"]) for u in rows]
        if len(rows) != expected or len(set(keys)) != len(keys):
            errors.append(block+" original phase coverage is incomplete or duplicated")
        observed_passes = sum(bool(p.get("passed")) for u in rows for p in u.get("points", []))
        observed_failures = sum(not p.get("passed") for u in rows for p in u.get("points", []))
        if (report.get("expected_phases") != expected
                or report.get("expected_points") != expected*6
                or report.get("passed_points") != observed_passes
                or report.get("failed_points") != observed_failures
                or report.get("unexecuted_points") != expected*6-observed_passes-observed_failures):
            errors.append(block+" reported counters contradict its preserved points")
        for key, row in zip(keys, rows, strict=True):
            if key in native_units:
                errors.append("phase was duplicated across native model families: "+str(key))
            native_units[key] = row
    if set(native_units) != set(identities):
        errors.append("native phase identities differ from the complete independent catalogue")
    comparisons = 0
    units = {}
    for case in cases:
        key = case["database"], case["phase"]
        row = {**case, "status": "failed", "numerical_passed_points": 0,
               "executed_points": 0, "failures": []}
        result["units"].append(row)
        source = native_units.get(key)
        if source is None:
            row["failures"].append("native phase was not executed or recorded")
            continue
        points = source.get("points", [])
        row["executed_points"] = len(points)
        if case["expected"] == "source_refusal":
            kinds = {issue["kind"] for issue in case["source_issues"]}
            expected_error = ("ValueError: MQMQA pair source atoms"
                if "pair_element_identity" in kinds else
                "ValueError: MQMQA quadruplet coordination violates charge")
            endings = [s.rstrip().splitlines()[-1]
                       for s in source.get("failures", []) if s.strip()]
            if (points or source.get("passed") or not case["source_issues"]
                    or not any(s.startswith(expected_error) for s in endings)):
                row["failures"].append(
                    "original invalid source was not refused for its independent cause")
            else:
                row["status"] = "source_refusal_confirmed"
            continue
        expected_points = {(profile, t) for profile in ("uniform", "tilted")
                           for t in case["temperatures"]}
        observed_points = {(p.get("profile"), p.get("T")) for p in points}
        if (case["expected"] != "numerical" or case["source_issues"]
                or not source.get("passed") or source.get("failures") or len(points) != 6
                or observed_points != expected_points
                or not all(p.get("passed") and not p.get("failures") and p.get("values")
                           for p in points)):
            row["failures"].append(
                "valid original phase has incomplete or failed numerical evidence")
            continue
        row.update(status="numerically_qualified", numerical_passed_points=6)
        row["comparisons"] = sum(len(p["values"]) for p in points)
        comparisons += row["comparisons"]
        for point in points:
            units.update({label: quantity_unit(label) for label in point["values"]})
    qualified = sum(u["status"] == "numerically_qualified" for u in result["units"])
    refused = sum(u["status"] == "source_refusal_confirmed" for u in result["units"])
    executed = sum(u["executed_points"] for u in result["units"])
    failed = sum(u["status"] == "failed" for u in result["units"])
    result.update(valid_phases_qualified=qualified, source_refusals_confirmed=refused,
        executed_original_points=executed, qualified_valid_points=qualified*6,
        unexecuted_original_points=max(0, 684-executed),
        extra_executed_points=max(0, executed-684), comparison_count=comparisons,
        failed_phase_judgments=failed, quantity_units=units,
        all_original_phases_numerically_qualified=qualified == 114,
        all_original_sources_physics_consumable=refused == 0 and qualified == 114,
        complete_catalogue_qualified=(not errors and qualified == 111 and refused == 3
                                     and failed == 0 and executed == 666))
    return result


def run_stage(name, command, log_path, completed, total):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    started = time.perf_counter()
    with (log_path.open("x", encoding="utf-8") as log,
          subprocess.Popen(command, stdout=log, stderr=log, env=environment) as process):
        while True:
            try:
                process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                print(f"catalogue stages {completed}/{total} active={name} "
                      f"alive_s={time.perf_counter()-started:.1f} bytes={log_path.stat().st_size} "
                      f"last={lines[-1] if lines else 'startup'}", flush=True)
    print(f"catalogue stages {completed+1}/{total} {name} exit={process.returncode}", flush=True)
    return process.returncode


def read_report(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-catalogue", action="store_true")
    args = parser.parse_args()
    if args.reference_catalogue:
        reference_catalogue(args.source_directory, args.output)
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("qualification data must remain outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {name: hashlib.sha256((args.source_directory/name).read_bytes()).hexdigest()
              for name in FILES}
    write(args.output/"plan.json", {"source_sha256": hashes, "expected_phases": 114,
        "expected_original_points": 684, "expected_valid_points": 666,
        "expected_source_refusals": 3, "reference_version": "0.11.2"})
    common = ["--source-directory", str(args.source_directory),
              "--reference-python", str(args.reference_python)]
    directory = Path(__file__).resolve().parent
    stages = (
        ("catalogue", [str(args.reference_python), "-I", str(Path(__file__).resolve()),
                       "--reference-catalogue", "--source-directory", str(args.source_directory),
                       "--output", str(args.output/"catalogue.json")]),
        ("structure", [sys.executable, str(directory/"run_chem_sage_input_block.py"),
                       *common, "--output", str(args.output/"structure")]),
        ("cef", [sys.executable, str(directory/"run_chem_sage_cef_block.py"),
                 *common, "--output", str(args.output/"cef")]),
        ("mqmqa", [sys.executable, str(directory/"run_mqmqa_block.py"),
                   *common, "--output", str(args.output/"mqmqa")]),
    )
    exits = {}
    for i, (name, command) in enumerate(stages):
        exits[name] = run_stage(name, command, args.output/(name+".log"), i, len(stages))
        write(args.output/"stage_exits.json", exits)
    try:
        result = assess(read_report(args.output/"catalogue.json"),
            read_report(args.output/"structure/summary.json"),
            read_report(args.output/"cef/native.json"),
            read_report(args.output/"mqmqa/native.json"), hashes)
        if exits != {"catalogue": 0, "structure": 0, "cef": 0, "mqmqa": 1}:
            result["failures"].append(
                "stage exits do not match complete execution with source refusals")
            result["complete_catalogue_qualified"] = False
        current = {name: hashlib.sha256((args.source_directory/name).read_bytes()).hexdigest()
                   for name in FILES}
        if current != hashes:
            result["failures"].append("source files changed while qualification was running")
            result["complete_catalogue_qualified"] = False
    except Exception:
        result = {"complete_catalogue_qualified": False, "expected_phases": 114,
                  "expected_original_points": 684, "failures": [traceback.format_exc()]}
    result["stage_exits"] = exits
    result["original_numerical_subblock_passed"] = {
        name: read_report(args.output/name/"summary.json").get("complete_and_passed", False)
        for name in ("cef", "mqmqa")}
    result["evidence_directory"] = str(args.output)
    write(args.output/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"units", "quantity_units"}}),
          flush=True)
    return 0 if result["complete_catalogue_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
