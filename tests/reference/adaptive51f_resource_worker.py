"""One resource condition/engine, with separate timed and allocation observations."""

import argparse
import json
import sys
import time
import traceback
import tracemalloc
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive51f_distribution_cases import author
from adaptive51f_resource_cases import SEEDS

from chemworld.physchem.stochastic_adaptive_advance import (
    advance_adaptive_hybrid,
    initial_adaptive_continuation,
)
from chemworld.physchem.stochastic_adaptive_hybrid import MOLECULAR_QUANTUM
from chemworld.physchem.stochastic_kinetics import advance_stochastic, initial_continuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--engine", required=True, choices=("direct", "next", "RK4", "LSODA"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    network, spec, amounts = author(
        {**unit, "method": args.engine if args.engine in {"RK4", "LSODA"} else "RK4"}
    )
    adaptive = args.engine in {"RK4", "LSODA"}
    if not adaptive:
        algorithm = (
            "gillespie-direct-v1" if args.engine == "direct" else "gibson-bruck-next-reaction-v1"
        )
        network = replace(network, stochastic={"volume_L": spec.volume_L, "algorithm": algorithm})
        counts = dict(unit["initial"])
    args.output.mkdir(parents=True, exist_ok=True)
    trials = [(seed, False) for seed in SEEDS] + [(17, True)]
    with (
        (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream,
        (args.output / "events.jsonl").open("x", encoding="utf-8") as journal,
    ):
        for index, (seed, allocation) in enumerate(trials):
            identity = unit["id"] + "-" + args.engine + "-" + str(index)
            row = {"id": identity, "seed": seed, "allocation_trial": allocation, "status": "error"}
            try:
                memory = (
                    initial_adaptive_continuation(network, spec, amounts, seed=seed)
                    if adaptive
                    else initial_continuation(seed, algorithm=algorithm)
                )
                journal.write(json.dumps({"id": identity, "event": "time_course_started"}) + "\n")
                journal.flush()
                if allocation:
                    tracemalloc.start()
                wall, cpu = time.perf_counter(), time.process_time()
                try:
                    result = (
                        advance_adaptive_hybrid(network, spec, amounts, memory, unit["duration_s"])
                        if adaptive
                        else advance_stochastic(network, counts, memory, unit["duration_s"])
                    )
                finally:
                    row.update(wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu)
                    if allocation:
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        row.update(
                            python_allocation_current_bytes=current,
                            python_allocation_peak_bytes=peak,
                        )
                row.update(
                    status="ok",
                    time_s=result.continuation["time_s"],
                    populations={k: v / MOLECULAR_QUANTUM for k, v in result.amounts_mol.items()}
                    if adaptive
                    else result.counts,
                    amounts_mol=result.amounts_mol
                    if adaptive
                    else {k: v * MOLECULAR_QUANTUM for k, v in result.counts.items()},
                    events=len(result.events),
                    work=result.work,
                    rng_state=result.continuation["rng_state"],
                    reaction_rate_visits=result.work["propensity_evaluations"]
                    + (
                        len(network.reactions) * result.work["drift_evaluations"] if adaptive else 0
                    ),
                )
                if unit["condition"] == "low":
                    row["event_path"] = result.events
            except Exception as exc:
                row.update(error=repr(exc), traceback=traceback.format_exc())
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "unit": unit["id"],
                        "engine": args.engine,
                        "completed": index + 1,
                        "total": 9,
                        "allocation_trial": allocation,
                        "status": row["status"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
