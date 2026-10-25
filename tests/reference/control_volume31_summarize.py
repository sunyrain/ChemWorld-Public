"""Generate the parent 31 report from retained actual-world outputs and test XML."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def functional(path):
    suite = ET.parse(path).getroot().find("testsuite")
    return {
        "source": path.name,
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "elapsed_s": float(suite.attrib["time"]),
        "test_names": [case.attrib["name"] for case in suite.findall("testcase")],
    }


def run(root, accepted, coordinator_xml, output):
    attempts = [
        {"attempt": path.parent.name, **json.loads(path.read_text(encoding="utf-8"))}
        for path in sorted(root.glob("*/summary.json"))
    ]
    summary = next(a for a in attempts if a["attempt"] == accepted)
    units = []
    for path in sorted((root / accepted).glob("[0-9][0-9]-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        last = record["checkpoints"][-1]
        solvers = [
            t["info"]["outcome_diagnostics"]["local_outcomes"]["control"]["solver"]
            for t in record["transactions"]
        ]
        measured = [m for group in record["comparisons"] for m in group.get("measurements", [])]
        units.append(
            {
                "unit": record["unit"],
                "card": record["card"],
                "reference_checkpoints": record["reference"],
                "final_native_amounts": last["amounts"],
                "final_ledger": last["ledger"],
                "final_native_rates_mol_s": last["rates_mol_s"],
                "native_original_wait_count": len(record["transactions"]),
                "native_original_wait_solver_costs": {
                    k: sum(s[k] for s in solvers) for k in ("nfev", "njev", "nlu", "solver_calls")
                },
                "in_process_replay": record["in_process_replay"]["snapshot"] == last["snapshot"],
                "fresh_process_exit": record["child_process"]["returncode"],
                "measurement_rows": len(measured),
                "maximum_numeric_tolerance_ratio": max(
                    m.get("maximum_tolerance_ratio", 0) for m in measured
                ),
            }
        )
    report = {
        "schema": "chemworld-parent31-control-volume-closure-v1",
        "claim": "36072a15e",
        "evidence_class": "synthetic-development",
        "accepted_attempt": accepted,
        **{
            k: summary[k]
            for k in ("units", "comparisons", "units_passed", "comparisons_passed", "elapsed_s")
        },
        "unexecuted_comparisons": sum(
            c["name"].startswith("unexecuted-slot")
            for unit in summary["unit_results"]
            for c in unit["comparisons"]
        ),
        "raw_artifacts": "%TEMP%/ChemWorld-31-close",
        "attempts": attempts,
        "accepted_units": units,
        "native_original_waits": sum(u["native_original_wait_count"] for u in units),
        "in_process_replay_waits": 4,
        "fresh_process_replay_waits": 4,
        "native_original_wait_solver_costs": {
            k: sum(u["native_original_wait_solver_costs"][k] for u in units)
            for k in ("nfev", "njev", "nlu", "solver_calls")
        },
        "solver_cost_scope": (
            "Counts cover the 16 original world waits; "
            "replay costs are not included or represented as zero."
        ),
        "reference_method": (
            "Independent closed analytic states, derivatives, port integrals and scalar h/u; "
            "no numerical reference integration or production evaluator."
        ),
        "current_consumer_groups": [
            functional(root / "serial-consumers.xml"),
            functional(coordinator_xml),
        ],
        "production_changes": [],
        "limitations": (
            "Declared major generic unit families; no full IDAES version parity, arbitrary "
            "coexistence, all reaction/EOS families or formal release qualification."
        ),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "units_passed",
                    "comparisons_passed",
                    "elapsed_s",
                    "native_original_wait_solver_costs",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--coordinator-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.runs_dir, args.accepted, args.coordinator_xml, args.output)
