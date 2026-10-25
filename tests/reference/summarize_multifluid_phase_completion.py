"""Retain every 27B completion attempt, unresolved unit and numerical check."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def stability_summary(raw):
    return {
        "reference_basin": raw["reference_basin"], "complete": raw["complete"],
        "unstable": raw["unstable"], "minimum_tpd_RT": raw["minimum_tpd_RT"],
        "failure_reason": raw["failure_reason"],
        "failed_roots": [r for r in raw["root_attempts"] if not r["success"]],
        "failed_starts": [s for s in raw["starts"] if not s["success"]],
    }


def phase_attempt(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    summary = {k: v for k, v in raw.items() if k not in {"units", "comparisons"}}
    summary["raw_output"] = str(path)
    summary["native_two_phase_units"] = sum(
        u.get("phase_resolved", False) and u["native"]["flash"]["phase_status"] == "two_phase"
        for u in raw["units"]
    )
    summary["units"] = []
    for unit in raw["units"]:
        outcome = {k: v for k, v in unit.items() if k not in {"native", "source_sha256"}}
        native = unit.get("native")
        if native:
            flash = native["flash"]
            outcome["native"] = {
                "success": native["success"], "failure_reason": native["failure_reason"],
                "phase_status": flash["phase_status"],
                "stability": stability_summary(flash["stability"]),
                "post_split_stability": [stability_summary(s)
                                         for s in flash["post_split_stability"]],
                "material_balance_error_mol": flash["material_balance_error_mol"],
                "chemical_potential_error_RT": flash["chemical_potential_error_RT"],
                "pressure_relative_error": flash["pressure_relative_error"],
                "solver_attempts": flash["solver_attempts"],
                "installed_phase_ids": native["installed_phase_ids"],
                "phase_enthalpies_J": native["phase_enthalpies_J"],
                "total_enthalpy_J": native["total_enthalpy_J"],
                "phases": [{k: v for k, v in phase.items() if k != "state"}
                           | {"density_mol_m3": phase["state"]["molar_density_mol_m3"],
                              "composition": phase["state"]["mole_fractions"],
                              "pressure_Pa": phase["state"]["pressure_Pa"]}
                           for phase in flash["phases"]],
            }
        summary["units"].append(outcome)
    summary["comparisons"] = raw["comparisons"]
    return summary


def junit(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    return {
        "raw_junit": str(path),
        **{key: sum(int(s.get(key, 0)) for s in suites)
           for key in ("tests", "failures", "errors", "skipped")},
        "elapsed_s": sum(float(s.get("time", 0)) for s in suites),
        "failed_tests": [{"name": c.get("name"), "class": c.get("classname"),
                          "details": [f.text for f in c if f.tag in {"failure", "error"}]}
                         for c in root.iter("testcase")
                         if any(f.tag in {"failure", "error"} for f in c)],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-junit", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite a previous completion summary")
    report = {
        "task": "NATIVE-CONV-27B", "mode": "development", "date": "2026-10-25",
        "owner": "sunyrain-Codex-phase27b", "claim": "286c09684",
        "shared_density_fix_claim": "0808bbdcb",
        "prior_evidence": "MULTIFLUID_CATALOG_RESULTS.json; unchanged",
        "final_phase_attempt": 2,
        "phase_attempts": [phase_attempt(args.raw_dir / name) for name in
                           ("default-attempt1.json", "local-attempt1.json",
                            "default-attempt2.json", "local-attempt2.json")],
        "caloric_attempts": [json.loads((args.raw_dir / name).read_text())
                             for name in ("caloric-attempt1.json", "caloric-attempt2.json")],
        "functional_attempts": [junit(args.raw_dir / name) for name in
                                ("pressure-regression-before.xml", "pressure-regression-after.xml",
                                 "common-regression-attempt1.xml",
                                 "source-consumer-regression.xml",
                                 "common-regression-attempt2.xml")],
        "focused_validation": {
            "ruff_log": str(args.raw_dir / "ruff-final.log"),
            "mypy_command": "uv run --no-sync mypy src/chemworld/physchem/molecular_eos.py "
                            "--follow-imports=silent",
            "mypy_result": "Success: no issues found in 1 source file",
            "distinct_functional_tests_latest": 142,
            "overlap": "Final common regression runs all 142 distinct cases together. "
                       "The previous 140 and separate 7 overlap; do not add their counts. "
                       "Initial 1/7 failed pressure trial is retained.",
        },
        "installed_probe": {
            "path": "tests/reference/probe_multifluid_phase_consumers.py",
            "fixture": str(args.raw_dir / "source-catalog.json"),
            "expected_cases": 2,
            "status": "awaiting coordinator isolated sdist-wheel run",
        },
        "acceptance_scope": {
            "source_catalogue": "prior 163 models, 652 states and original domain/identity checks",
            "actual_two_phase_consumption": "2 original units per complete default/local block",
            "all_requested_phases_resolved": False,
            "retained_unresolved": "9 methane/ethane requests in each original block",
            "global_phase_stability": False,
            "empirical_material_accuracy": False,
            "full_GERG_catalogue": False,
        },
    }
    if args.installed_junit is not None:
        installed = junit(args.installed_junit)
        passed = installed["tests"] == 1 and not any(
            installed[key] for key in ("failures", "errors", "skipped")
        )
        report["installed_probe"].update(
            status="passed" if passed else "failed", qualification=installed,
            command="uv run --no-sync --with fluids==1.3.1 --with networkx==3.6.1 "
                    "--with teqp==0.23.2 pytest -q tests/test_atomic_packaging.py --tb=short",
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
