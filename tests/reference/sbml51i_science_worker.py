"""Complete original SBML integer ensembles or separately measured advance trials."""

import argparse
import json
import platform
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from sbml51i_science_cases import ENCODINGS, STRATEGIES, TRIALS, author, seed

    from chemworld.physchem.sbml_stochastic import initial_memory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--encoding", choices=ENCODINGS, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--mode", choices=("distribution", "resource"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    spec = author(unit, args.encoding, args.strategy)
    (args.output / "model.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "mode": args.mode,
                "timed_scope": "original_sbml_integer_advance_excluding_xml_and_io",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    trials = (
        TRIALS
        if args.mode == "resource"
        else [(seed(unit, args.strategy, i), "distribution") for i in range(unit["replicates"])]
    )
    started = last = time.monotonic()
    with (
        (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream,
        (args.output / "events.jsonl").open("x", encoding="utf-8") as journal,
    ):
        for index, (random_seed, purpose) in enumerate(trials):
            row = {
                "id": unit["id"] + "-" + str(index),
                "seed": random_seed,
                "purpose": purpose,
                "simulation_attempted": False,
                "simulation_completed": False,
                "status": "error",
            }
            try:
                memory = initial_memory(random_seed, spec.solver_method)
                journal.write(json.dumps({"id": row["id"], "event": "time_course_started"}) + "\n")
                journal.flush()
                row["simulation_attempted"] = True
                if purpose == "allocation":
                    tracemalloc.start()
                wall, cpu = time.perf_counter_ns(), time.process_time_ns()
                try:
                    values, continuation, result = spec.integer_kinetics.advance(
                        spec.frame.initial_values,
                        memory,
                        0.0,
                        unit["duration_s"],
                        spec.solver_method,
                        spec.maximum_events,
                        spec.tau_policy,
                    )
                finally:
                    row.update(
                        wall_s=(time.perf_counter_ns() - wall) / 1e9,
                        cpu_s=(time.process_time_ns() - cpu) / 1e9,
                    )
                    if purpose == "allocation":
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        row.update(
                            python_allocation_current_bytes=current,
                            python_allocation_peak_bytes=peak,
                        )
                row.update(
                    status="ok",
                    simulation_completed=True,
                    coordinates=values,
                    counts=spec.integer_kinetics.counts(values, unit["duration_s"]),
                    time_s=continuation["process"]["time_s"],
                    continuation=continuation,
                    committed_firings=continuation["process"]["events_committed"],
                    events=len(result.events),
                    batches=len(result.batches),
                    work=result.work,
                )
            except Exception as exc:
                row.update(error=repr(exc), traceback=traceback.format_exc())
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            now = time.monotonic()
            if now - last >= 30 or index + 1 == len(trials):
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "completed": index + 1,
                            "total": len(trials),
                            "rows_per_second": (index + 1) / elapsed if elapsed else None,
                            "eta_s": (len(trials) - index - 1) * elapsed / (index + 1),
                        }
                    ),
                    flush=True,
                )
                last = now


if __name__ == "__main__":
    main()
