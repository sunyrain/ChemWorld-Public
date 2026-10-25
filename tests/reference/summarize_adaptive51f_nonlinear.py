"""Every nonlinear attempt, complete denominators and compact failures with raw provenance."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51f-nonlinear-*")):
        if not directory.is_dir():
            continue
        if not (directory / "summary.json").is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for unit in data["units"]:
            for engine, summary in unit["engines"].items():
                summary["failures"] = [
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        in {
                            "id",
                            "seed",
                            "status",
                            "error",
                            "stage",
                            "simulation_attempted",
                            "simulation_completed",
                            "partial_recorded_steps",
                            "process_returned",
                            "partial_capture_error",
                            "last_partial_time_s",
                        }
                    }
                    for row in summary["failures"]
                ]
                if engine == "copasi":
                    versions, methods, partial_times = set(), set(), set()
                    completed_courses = returned_false = partial_paths = 0
                    with (directory / unit["id"] / engine / "trajectories.jsonl").open(
                        encoding="utf-8"
                    ) as stream:
                        for line in stream:
                            row = json.loads(line)
                            if "version" in row:
                                versions.add(row["version"])
                            if "method_name" in row:
                                methods.add(row["method_name"])
                            completed_courses += bool(row.get("simulation_completed", False))
                            returned_false += row.get("process_returned") is False
                            partial_paths += "partial_rows" in row
                            if row.get("last_partial_time_s") is not None:
                                partial_times.add(row["last_partial_time_s"])
                    unit["source_versions"], unit["selected_methods"] = (
                        sorted(versions),
                        sorted(methods),
                    )
                    summary.update(
                        time_courses_reported_completed=completed_courses,
                        explicitly_reported_false_returns=returned_false,
                        partial_trajectories_retained=partial_paths,
                        partial_last_times_s=sorted(partial_times),
                    )
                elif engine == "native":
                    rejected = 0
                    with (directory / unit["id"] / engine / "trajectories.jsonl").open(
                        encoding="utf-8"
                    ) as stream:
                        for line in stream:
                            row = json.loads(line)
                            trace = row.get("traceback", "")
                            if (
                                row["status"] != "ok"
                                and "initial_adaptive_continuation" in trace
                                and "compile_elementary_channels" in trace
                            ):
                                rejected += 1
                    summary["pre_advance_admission_rejections"] = rejected
                    # First runner journaled entry before continuation admission;
                    # later explicit flags journal actual advance starts only.
                    old_labels = any("simulation_attempted" not in r for r in summary["failures"])
                    summary["verified_numeric_advance_starts"] = (
                        summary["simulations_attempted"] - rejected
                        if old_labels
                        else summary["simulations_attempted"]
                    )
        totals = {}
        for engine in ("native", "reference", "copasi"):
            totals[engine] = {
                key: sum(u["engines"][engine][key] for u in data["units"])
                for key in (
                    "planned",
                    "recorded",
                    "completed",
                    "simulations_attempted",
                    "unstarted",
                    "unrecorded",
                    "violating_points",
                )
            }
            totals[engine]["max_pool_error_particles"] = max(
                u["engines"][engine]["max_pool_error_particles"] for u in data["units"]
            )
        totals["native"]["pre_advance_admission_rejections"] = sum(
            u["engines"]["native"]["pre_advance_admission_rejections"] for u in data["units"]
        )
        totals["native"]["verified_numeric_advance_starts"] = sum(
            u["engines"]["native"]["verified_numeric_advance_starts"] for u in data["units"]
        )
        attempts.append(
            {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                **data,
                "totals": totals,
                "native_passed_units": sum(u["native_passed"] for u in data["units"]),
                "source_passed_units": sum(u["source_passed"] for u in data["units"]),
            }
        )
    result = {
        "schema": "chemworld-adaptive51f-nonlinear-history-v1",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "scope": "Nonlinear finite-grid physics and actual source; source policy/errors retained",
        "full_source_parameters_rng_partial_outputs_and_tracebacks_retained_outside_git": True,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
