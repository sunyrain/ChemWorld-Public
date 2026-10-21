"""Full six-case official source block for volume/concentration event semantics."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.coordinate_hybrid import CoordinateHybridMemory, advance_coordinate_events
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_events import SBMLTriggerRoots, sbml_coordinate_events
from chemworld.physchem.sbml_kinetics import SBMLKinetics, sbml_continuous_system
from chemworld.physchem.solver_backend import ODESolverPolicy

CASES = ("01506", "01507", "01508", "01509", "01779", "01780")
COMMIT = "d6dcd2cc6408d5eb881638ac6ce7a05fd7d2e4d1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if any(p.resolve().is_relative_to(root) for p in (args.source_directory, args.output)):
        raise ValueError("scientific inputs and outputs must stay outside Git")
    catalog = json.loads((args.source_directory/"tree.json").read_text(encoding="utf-8"))
    if catalog["sha"] != COMMIT or catalog["truncated"]:
        raise ValueError("official source catalogue differs from the fixed commit")
    args.output.mkdir(exist_ok=False)
    report = {"expected_inputs": 6, "expected_comparisons": 198, "units": [],
        "source_commit": COMMIT, "executed_inputs": 0,
        "explicit_unit_policy": {"timeUnits": "second", "substanceUnits": "mole",
                                 "extentUnits": "mole", "volumeUnits": "litre"}}

    def write():
        (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")

    started = time.perf_counter()
    for case in CASES:
        unit = {"input": case, "passed": False, "failures": [], "comparisons": []}
        report["units"].append(unit)
        try:
            folder = args.source_directory/"cases/semantic"/case
            source = InputSource(DataSourceProvenance("official-sbml-"+case, 0,
                "SBML Test Suite semantic "+case, COMMIT,
                "not redistributed; licence retained in original archive"), "literature")
            model = SBMLModel.from_file(folder/(case+"-sbml-l3v2.xml"), source=source,
                                       unit_defaults=report["explicit_unit_policy"])
            with (folder/(case+"-results.csv")).open(encoding="utf-8-sig", newline="") as stream:
                expected = list(csv.DictReader(stream))
            if len(expected) != 11 or set(expected[0]) != {"time", "C1", "S1", "x"}:
                raise ValueError("original official node/quantity coverage differs")
            frame = SBMLKinetics(model)
            events, roots = sbml_coordinate_events(frame), SBMLTriggerRoots(frame)
            policy = ODESolverPolicy("sbml_official_bdf", "BDF", rtol=1e-10, atol=1e-12,
                                     max_step_s=.01)
            system = sbml_continuous_system(frame, policy=policy)
            state = CoupledState({}, 298.15, 1., coordinates=frame.initial_values)
            memory = CoordinateHybridMemory.initialize(events)
            unit["events"], unit["trajectory"], unit["segment_diagnostics"] = [], [], []
            for row in expected:
                target = float(row["time"])
                result = advance_coordinate_events(system, state, target-state.time_s, memory,
                    events, roots=roots.roots, trigger_values=roots.trigger_values)
                state, memory = result.final_state, result.memory
                amounts, volumes = frame.quantities(state.coordinates, state.time_s)
                values = {"C1": volumes["C1"], "S1": amounts["S1"],
                          "x": frame.context(state.coordinates, state.time_s)["x"]}
                unit["trajectory"].append({"time_s": state.time_s, "values": values,
                    "coordinates": dict(state.coordinates)})
                unit["events"].extend(result.events)
                unit["segment_diagnostics"].extend(result.segment_diagnostics)
                for key, value in values.items():
                    reference = float(row[key])
                    absolute, relative = ((1e-12, 2e-8) if key == "C1" else
                                           (2e-10, 2e-6) if key == "S1" else (2e-8, 2e-6))
                    tolerance = absolute+relative*abs(reference)
                    comparison = {"time_s": target, "quantity": key, "actual": value,
                        "reference": reference, "error": abs(value-reference),
                        "tolerance": tolerance, "passed": abs(value-reference) <= tolerance}
                    unit["comparisons"].append(comparison)
                    if not comparison["passed"]:
                        unit["failures"].append(comparison)
            report["executed_inputs"] += 1
            unit["passed"] = not unit["failures"]
        except Exception:
            unit["failures"].append(traceback.format_exc())
        write()
        n = len(report["units"])
        elapsed = max(time.perf_counter()-started, 1e-9)
        print(f"official {n}/6 rate={n/elapsed:.2f}/s eta={(6-n)*elapsed/n:.1f}s "
              f"passed={unit['passed']}", flush=True)
    report["passed_inputs"] = sum(u["passed"] for u in report["units"])
    report["unexecuted_inputs"] = 6-report["executed_inputs"]
    report["comparison_count"] = sum(len(u["comparisons"]) for u in report["units"])
    report["complete_and_passed"] = (report["passed_inputs"] == 6
                                      and report["comparison_count"] == 198)
    write()
    print(json.dumps({k: v for k, v in report.items() if k != "units"}), flush=True)
    return 0 if report["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
