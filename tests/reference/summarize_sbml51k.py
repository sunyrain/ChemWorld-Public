"""Fixed development histories, with optional actual installation attempts."""

import argparse
import json
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    attempts = {}
    for label in (
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
    ):
        directory = args.evidence / ("conv51k-" + label)
        data = read(directory / "summary.json")
        attempts[label] = {
            **data,
            **junit(directory / "junit.xml"),
            "raw_directory": str(directory),
            "complete_new_module_source_capture": label
            in {
                "integrated-second",
                "events-fourth",
                "post-merge-functional-first",
                "events-fifth",
            },
        }
    save("FUNCTIONAL_RESULTS.json", attempts)
    histories = {}
    for phase in ("before", "after", "post-merge"):
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
            suffixes = data.get("exact_suffixes", len(data["units"]))
            rows.append(
                {
                    "kind": kind,
                    "planned_suffixes": planned,
                    "recorded_cases": len(data["units"]),
                    "exact_suffixes": suffixes,
                    "passed": data["passed"] and suffixes == planned,
                    "elapsed_s": data["elapsed_s"],
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
    print(
        json.dumps(
            {"written": 3 + bool(args.package_attempt), "output": str(args.output),
             "scope": "development checks; independent science/source qualification separate"}
        )
    )


if __name__ == "__main__":
    main()
