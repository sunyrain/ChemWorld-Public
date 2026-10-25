"""Isolated original integer advance timing, with separate warm-up/allocation."""

import argparse
import json
import platform
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integer51h_distribution_cases import STRATEGIES, author

from chemworld.physchem.stochastic_kinetics import advance_stochastic, initial_continuation

TRIALS = (
    [(997, "warmup")]
    + [(s, "timed") for s in (1, 7, 17, 37, 73, 101, 251, 509)]
    + [(17, "allocation")]
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    network = author(unit, args.strategy)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "strategy": args.strategy,
                "model": network.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.monotonic()
    with (
        (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream,
        (args.output / "events.jsonl").open("x", encoding="utf-8") as journal,
    ):
        for index, (seed, purpose) in enumerate(TRIALS):
            identity = unit["id"] + "-" + str(index)
            row = {
                "id": identity,
                "seed": seed,
                "purpose": purpose,
                "status": "error",
                "simulation_attempted": False,
                "simulation_completed": False,
            }
            try:
                memory = initial_continuation(seed, algorithm=network.stochastic["algorithm"])
                journal.write(json.dumps({"id": identity, "event": "time_course_started"}) + "\n")
                journal.flush()
                row["simulation_attempted"] = True
                if purpose == "allocation":
                    tracemalloc.start()
                wall, cpu = time.perf_counter(), time.process_time()
                try:
                    result = advance_stochastic(
                        network, unit["initial"], memory, unit["duration_s"]
                    )
                finally:
                    row.update(wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu)
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
                    counts=result.counts,
                    time_s=result.continuation["time_s"],
                    committed_firings=result.continuation["events_committed"],
                    events=len(result.events),
                    batches=len(result.batches),
                    work=result.work,
                    continuation=result.continuation,
                )
            except Exception as exc:
                row.update(error=repr(exc), traceback=traceback.format_exc())
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "unit": unit["id"],
                        "strategy": args.strategy,
                        "completed": index + 1,
                        "total": len(TRIALS),
                        "trials_per_second": (index + 1) / elapsed if elapsed else None,
                        "eta_s": (len(TRIALS) - index - 1) * elapsed / (index + 1),
                        "purpose": purpose,
                        "status": row["status"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
