"""One fixed G engine/unit, with physical start flags and flushed failure records."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
AVOGADRO = 6.02214076e23


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--engine", choices=("native", "reference"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "native":
        from hybrid51g_distribution_cases import author

        from chemworld.physchem.stochastic_hybrid import (
            advance_stochastic_hybrid,
            initial_hybrid_continuation,
        )

        network, spec, amounts = author(unit)
        (args.output / "model.json").write_text(
            json.dumps({"network": network.to_dict(), "spec": spec.to_dict()}, indent=2),
            encoding="utf-8",
        )

        def simulate(seed, on_start):
            memory = initial_hybrid_continuation(network, spec, amounts, seed=seed)
            on_start()
            result = advance_stochastic_hybrid(network, spec, amounts, memory, unit["duration_s"])
            return {
                "populations": {k: value * AVOGADRO for k, value in result.amounts_mol.items()},
                "time_s": result.continuation["time_s"],
                "time_basis": "native_continuation",
                "events": len(result.events),
                "work": result.work,
                "rng_state": result.continuation["rng_state"],
            }

        base = 512600000
    else:
        if unit["kind"] == "feedback":
            from stochastic_hybrid51_reference import simulate as closed
        else:
            from cross_partition51_reference import simulate as closed
        assert not any(k == "chemworld" or k.startswith("chemworld.") for k in sys.modules)

        def simulate(seed, on_start):
            on_start()
            result = closed(unit["parameters"], seed, unit["duration_s"])
            if unit["kind"] == "feedback":
                total = unit["parameters"]["A0_mol"] * AVOGADRO
                values = {
                    "A": result["fraction_A"] * total,
                    "B": (1 - result["fraction_A"]) * total,
                    "E0": 1 - result["E1"],
                    "E1": result["E1"],
                }
            else:
                values = {k: result[k] for k in ("D", "A", "B")}
            return {
                "populations": values,
                "time_s": unit["duration_s"],
                "time_basis": "declared_analytic_horizon",
                "events": len(result["events"]),
                "last_event_time_s": max((e["time_s"] for e in result["events"]), default=None),
            }

        base = 512700000
    total = unit[args.engine + "_replicates"]
    started = last = time.monotonic()
    with (
        (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream,
        (args.output / "events.jsonl").open("x", encoding="utf-8") as journal,
    ):
        for replicate in range(total):
            identity = unit["id"] + "-" + str(replicate)
            seed = 17 if unit["frozen"] else base + 100000 * unit["index"] + replicate
            row = {
                "id": identity,
                "seed": seed,
                "status": "error",
                "stage": "continuation",
                "simulation_attempted": False,
                "simulation_completed": False,
            }
            journal.write(json.dumps({"id": identity, "event": "trajectory_started"}) + "\n")
            journal.flush()

            def on_start(row=row, identity=identity):
                row.update(stage="time_course", simulation_attempted=True)
                journal.write(json.dumps({"id": identity, "event": "time_course_started"}) + "\n")
                journal.flush()

            try:
                row.update(
                    simulate(seed, on_start), status="ok", stage="output", simulation_completed=True
                )
            except Exception as exc:
                row.update(error=repr(exc), traceback=traceback.format_exc())
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            now = time.monotonic()
            if now - last >= 30 or replicate + 1 == total:
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "unit": unit["id"],
                            "engine": args.engine,
                            "completed": replicate + 1,
                            "total": total,
                            "elapsed_s": elapsed,
                            "trajectories_per_second": (replicate + 1) / elapsed
                            if elapsed > 0
                            else None,
                            "eta_s": (total - replicate - 1) * elapsed / (replicate + 1),
                        }
                    ),
                    flush=True,
                )
                last = now


if __name__ == "__main__":
    main()
