"""Compact retained RED attempts, exact denominators and independent solver costs."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def run(root, accepted, output):
    attempts = []
    for directory in sorted(root.iterdir()):
        path = directory / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        attempts.append(
            {
                "attempt": directory.name,
                **{
                    k: data[k]
                    for k in (
                        "units",
                        "comparisons",
                        "units_passed",
                        "comparisons_passed",
                        "elapsed_s",
                    )
                },
                "failures": [
                    {
                        **failure,
                        "exception": str(failure["exception"])[:512]
                        if failure.get("exception")
                        else None,
                    }
                    for failure in data["failures"]
                ],
            }
        )
    functional = []
    for path in sorted(root.glob("functional*.xml")):
        suite = ET.parse(path).getroot().find("testsuite")
        functional.append(
            {
                "attempt": path.stem,
                "tests": int(suite.attrib["tests"]),
                "failures": int(suite.attrib["failures"]),
                "errors": int(suite.attrib["errors"]),
                "elapsed_s": float(suite.attrib["time"]),
                "failed_tests": [
                    case.attrib["name"]
                    for case in suite.findall("testcase")
                    if case.find("failure") is not None or case.find("error") is not None
                ],
            }
        )
    units = []
    for path in sorted((root / accepted).glob("[0-9][0-9]-*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        compact = {
            "unit": row["unit"],
            "passed": row["passed"],
            "comparisons": 10,
            "comparisons_passed": sum(c["passed"] for c in row["comparisons"]),
            "elapsed_s": row["elapsed_s"],
        }
        reference = row.get("reference", {})
        if "reference_costs" in reference:
            compact["reference_costs"] = reference["reference_costs"]
        elif "costs" in reference:
            compact["reference_costs"] = reference["costs"]
        elif "first" in reference:
            compact["reference_costs"] = {
                k: reference["first"]["costs"][k] + reference["second"]["costs"][k]
                for k in reference["first"]["costs"]
            }
        if "levels" in row:
            compact["grid_relative_errors"] = [
                {
                    k: level[k]
                    for k in ("segments", "terminal_relative_error", "profile_relative_error")
                }
                for level in row["levels"]
            ]
            compact["reference_costs"] = {
                k: sum(level["reference"]["reference_costs"][k] for level in row["levels"])
                for k in row["levels"][0]["reference"]["reference_costs"]
            }
        native = row.get("native", {})
        if "advance" in native:
            diag = native["advance"]["solver_diagnostic"]
            compact["native_costs"] = {
                k: diag[k]
                for k in ("pore_solves", "rhs_evaluations", "off_neutral_numerical_probes")
            }
            compact["physical_result"] = {
                k: native["receipt"][k]
                for k in (
                    "load_energy_delivered_J",
                    "bath_heat_into_liquid_J",
                    "free_energy_decrease_J",
                    "free_energy_decrease_minus_load_J",
                )
            }
            compact["actual_duration_s"] = native["advance"]["final_state"]["time_s"]
            compact["stopping_event"] = native["advance"]["stopping_event"]
        if "consumer" in row:
            compact["final_ledger"] = row["consumer"]["final_state"]["ledger"]
            compact["fresh_process_exit"] = row["fresh_process"]["returncode"]
        units.append(compact)
    final = next(r for r in attempts if r["attempt"] == accepted)
    result = {
        "schema": "chemworld-passive-red-development-v1",
        "evidence_class": "synthetic-development",
        "accepted_attempt": accepted,
        "units": 24,
        "comparisons": 240,
        "units_passed": final["units_passed"],
        "comparisons_passed": final["comparisons_passed"],
        "elapsed_s": final["elapsed_s"],
        "unexecuted_comparisons": sum(
            "unexecuted-slot" in f["comparison"]["name"] for f in final["failures"]
        ),
        "raw_artifacts": "%TEMP%/ChemWorld-48E",
        "attempts": attempts,
        "functional_attempts": functional,
        "accepted_unit_results": units,
        "reference_costs_total": {
            k: sum(r.get("reference_costs", {}).get(k, 0) for r in units)
            for k in ("bvp_calls", "bvp_nodes", "rhs_calls")
        },
        "scope": (
            "Original periodic interior C-AEM-D-CEM-C; ideal dilute current inventories; "
            "no terminal electrodes/full stack/nonideal friction or empirical calibration."
        ),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "units_passed",
                    "comparisons_passed",
                    "elapsed_s",
                    "reference_costs_total",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.runs_dir, args.accepted, args.output)
