"""Fixed development histories, with optional actual installation attempts."""

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def junit(path):
    root = ET.parse(path).getroot()
    counts = {
        key: sum(int(s.get(key, "0")) for s in root.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    failed, counters = [], []
    for node in root.iter("testcase"):
        errors = [*node.findall("failure"), *node.findall("error")]
        if "reported_kernel_work" in node.get("name", ""):
            counters.append({"node": node.get("name"), "passed": not errors})
        for error in errors:
            failed.append(
                {
                    "node": node.get("classname", "") + "::" + node.get("name", ""),
                    "message": error.get("message"),
                    "text": error.text,
                }
            )
    return {"counts": counts, "failed_nodes": failed, "kernel_counter_nodes": counters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-attempt", action="append", default=[])
    parser.add_argument("--science-attempt", action="append", default=[])
    parser.add_argument("--source-validity-attempt", action="append", default=[])
    parser.add_argument("--functional-attempt", action="append", default=[])
    parser.add_argument("--legacy-phase", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, data):
        (args.output / name).write_bytes(json.dumps(data, indent=2).encode("utf-8"))

    attempts = {}
    for label in (*(
        "boundaries-first",
        "boundaries-second",
        "events-first",
        "events-second",
        "events-third",
        "original-integration-first",
        "integrated-second",
        "events-fourth",
        "post-merge-functional-first",
        "events-fifth",
    ), *args.functional_attempt):
        directory = args.evidence / ("conv51k-" + label)
        data = read(directory / "summary.json")
        attempts[label] = {
            **data,
            **junit(directory / "junit.xml"),
            "raw_directory": str(directory),
            "complete_new_module_source_capture": label in args.functional_attempt
            or label in {
                "integrated-second",
                "events-fourth",
                "post-merge-functional-first",
                "events-fifth",
            },
        }
        origin = args.evidence / ("conv51k-" + label + "-origin.json")
        if origin.is_file():
            attempts[label]["package_origin_check_before_pytest"] = read(origin)
    save("FUNCTIONAL_RESULTS.json", attempts)
    histories = {}
    for phase in ("before", "after", "post-merge", *args.legacy_phase):
        rows = []
        for kind, planned in (
            ("formal", 24),
            ("integer", 24),
            ("experiment", 96),
            ("installed", 24),
        ):
            label = kind + "-legacy-" + phase
            if kind == "integer" and phase == "after":
                label = "integer-legacy-boundaries"
            directory = args.evidence / ("conv51k-" + label)
            data = read(directory / "summary.json")
            suffixes = data.get("exact_suffixes", sum(row["passed"] for row in data["units"]))
            rows.append(
                {
                    "kind": kind,
                    "planned_suffixes": planned,
                    "recorded_cases": len(data["units"]),
                    "exact_suffixes": suffixes,
                    "passed": data["passed"] and suffixes == planned,
                    "elapsed_s": data["elapsed_s"],
                    "elapsed_s_measured": data.get("elapsed_s_measured", True),
                    "unstarted_suffixes": data.get("unstarted_suffixes", 0),
                    "error": data.get("error"),
                    "raw_directory": str(directory),
                    "failures": [r for r in data["units"] if not r["passed"]],
                }
            )
        histories[phase] = {
            "blocks": rows,
            "planned_suffixes": 168,
            "exact_suffixes": sum(row["exact_suffixes"] for row in rows),
            "passed": all(row["passed"] for row in rows),
        }
    save("LEGACY_RESULTS.json", histories)
    diagnostics = {}
    for label in (
        "j-resume-diagnostic",
        "namespace-diagnostic-first",
        "namespace-diagnostic-second",
        "work-counter-first",
    ):
        path = args.evidence / ("conv51k-" + label + ".xml")
        diagnostics[label] = {**junit(path), "raw_junit": str(path)}
    for phase in ("before", "after"):
        directory = args.evidence / ("conv51k-namespace-source-" + phase)
        diagnostics["source-" + phase] = {
            **read(directory / "summary.json"),
            "raw_directory": str(directory),
        }
    policy = args.evidence / "conv51k-guard-policy-first/junit.xml"
    if policy.is_file():
        diagnostics["guard-policy-first"] = {**junit(policy), "raw_junit": str(policy),
            "scope": "pure candidate dependency helper; not connected to advancing kernels"}
    inspection_directory = args.evidence / "conv51k-pending-inspection-first"
    if (inspection_directory / "summary.json").is_file():
        inspection = read(inspection_directory / "summary.json")
        diagnostics["pending-inspection-first"] = {
            **{k: v for k, v in inspection.items() if k != "cells"},
            "raw_directory": str(inspection_directory),
            "positive_pending_leap_cells": [{
                "id": row["id"], "encoding": row["encoding"], "strategy": row["strategy"],
                "source_events_executed": row["source_events_executed"],
                "observed_A": row["observed_A"],
                "pending_after_horizon_count": len(row["positive_leap_packets_after_horizon"]),
                "pending_end_times_s": sorted({p["end_time_s"] for p in
                                               row["positive_leap_packets_after_horizon"]})}
                for row in inspection["cells"] if row["positive_leap_packets_after_horizon"]]}
    for label in ("event-tau-candidate-first", "event-tau-candidate-second",
                  "event-tau-candidate-third"):
        directory = args.evidence / ("conv51k-" + label)
        if (directory / "summary.json").is_file():
            diagnostics[label] = {
                **read(directory / "summary.json"), **junit(directory / "junit.xml"),
                "raw_directory": str(directory)}
    for label in ("event-tau-accuracy-first", "event-tau-accuracy-second"):
        directory = args.evidence / ("conv51k-" + label)
        if (directory / "summary.json").is_file():
            data = read(directory / "summary.json")
            diagnostics[label] = {
                **{k: v for k, v in data.items() if k != "cells"},
                "raw_directory": str(directory),
                "cells": [{**{k: v for k, v in cell.items() if k != "engine"},
                           "engine": {k: v for k, v in cell["engine"].items()
                                      if k not in {"normalized_grids"}}}
                          for cell in data["cells"]]}
    save("DIAGNOSTIC_RESULTS.json", diagnostics)
    if args.package_attempt:
        packages = {}
        for label in args.package_attempt:
            directory = args.evidence / ("conv51k-" + label)
            data = read(directory / "summary.json")
            probe = data.get("protocol_summary", {})
            rows = probe.get("protocols", [])
            packages[label] = {
                "raw_directory": str(directory),
                "passed": data["passed"],
                "planned_experiments": 12,
                "recorded_experiments": len(rows),
                "successful_experiments": sum(r["passed"] for r in rows),
                "experiments_without_final_record": 12-len(rows),
                "planned_courses": 36,
                "completed_courses_in_successful_experiments": sum(
                    r.get("courses", 0) for r in rows),
                "experiment_actions": sum(r.get("recorded_steps", 0) for r in rows),
                "experiment_replayed_actions": sum(r.get("replayed_steps", 0) for r in rows),
                "unique_native_suffix_actions": sum(r.get("native_suffix_steps", 0) for r in rows),
                "native_combined_replayed_actions_including_overlapping_prefixes": sum(
                    r.get("native_combined_replayed_steps", 0) for r in rows),
                "experiment_fresh_process_exact": sum(r.get("new_process_exact", False)
                                                      for r in rows),
                "native_fresh_process_exact": sum(r.get("native_new_process_exact", False)
                                                  for r in rows),
                "actual_native_campaigns": sum(r.get("native_sessions", 0) for r in rows),
                "stages": data["stages"],
                "unexecuted_stages": data["unexecuted_stages"],
                "artifacts": data.get("artifacts"),
                "production_dependencies": data.get("dependencies"),
                "sdk_importable": probe.get("sdk_importable"),
                "module_origins": probe.get("module_origins"),
                "protocols": rows,
                "failures": [r for r in rows if not r["passed"]],
                "error": data.get("error"),
                "traceback": data.get("traceback"),
                "probe_stderr": (directory / "probe.stderr").read_text(encoding="utf-8")
                    if (directory / "probe.stderr").is_file() else None,
                "elapsed_s": data["elapsed_s"],
            }
        save("PACKAGE_RESULTS.json", packages)
    if args.science_attempt:
        science = {}
        for label in args.science_attempt:
            directory = args.evidence / ("conv51k-" + label)
            data = read(directory / "summary.json")
            cells = []
            for unit in data["units"]:
                for cell in unit["cells"]:
                    engines = {}
                    for name, engine in cell["engines"].items():
                        engines[name] = {k: v for k, v in engine.items() if k not in {
                            "normalized_grids", "failures", "invalid_trajectories"}}
                        engines[name]["failures"] = [{k: row.get(k) for k in (
                            "id", "seed", "stage", "simulation_attempted", "simulation_completed",
                            "process_returned", "method_name", "error")}
                            for row in engine["failures"]]
                        engines[name]["invalid_trajectories"] = [
                            {"id": row["id"], "issues": [
                                {k: value for k, value in issue.items() if k != "raw"}
                                if isinstance(issue, dict) else issue for issue in row["issues"]]}
                            for row in engine["invalid_trajectories"]]
                    cells.append({"id": unit["id"], "family": unit["family"],
                        "encoding": cell["encoding"], "strategy": cell["strategy"],
                        "generator": cell["generator"], "passed": cell["passed"],
                        "engines": engines})
            science[label] = {k: v for k, v in data.items() if k not in {"units"}}
            science[label].update(raw_directory=str(directory), cells=cells,
                passed_generator_cells=sum(c["generator"]["passed"] for c in cells),
                passed_reference_laws=sum(u["reference_passed"] for u in data["units"]),
                reference_laws=[{"id": u["id"], "passed": u["reference_passed"],
                    "diagnostics": u["reference_diagnostics"], "process": u["reference_process"]}
                    for u in data["units"]])
            if data["mode"] == "wiring" and not data["passed"]:
                science[label]["original_downstream_blocks_not_started"] = {
                    "native_distribution": 163840, "source_distribution": 163840,
                    "native_resource_trials": 800,
                    "reason": "complete combined source wiring gate failed"}
        save("SCIENCE_RESULTS.json", science)
    if args.source_validity_attempt:
        validity = {}
        for label in args.source_validity_attempt:
            directory = args.evidence / ("conv51k-" + label)
            data = read(directory / "summary.json")
            probes = []
            for row in data["probes"]:
                probes.append({k: v for k, v in row.items() if k != "messages"})
                probes[-1]["trajectory_message_codes"] = re.findall(
                    r"CTrajectoryMethod \((\d+)\)", row.get("messages", ""))
            validity[label] = {**data, "probes": probes, "raw_directory": str(directory),
                "source_code_directory": str(args.evidence / "conv51k-source-code-inspection"),
                "primary_references": [
                    "https://raw.githubusercontent.com/copasi/COPASI/Build-309/copasi/trajectory/"
                    + name + ".cpp" for name in (
                        "CStochMethod", "CTauLeapMethod", "CStochDirectMethod")],
                "accepted": sum(r.get("valid_problem") is True for r in data["probes"]),
                "rejected": sum(r.get("valid_problem") is False for r in data["probes"])}
        save("SOURCE_VALIDITY_RESULTS.json", validity)
    print(
        json.dumps(
            {"written": 3 + bool(args.package_attempt) + bool(args.science_attempt)
                + bool(args.source_validity_attempt), "output": str(args.output),
             "scope": "development checks; independent science/source qualification separate"}
        )
    )


if __name__ == "__main__":
    main()
