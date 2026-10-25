"""One fixed engine/unit, flushing all trajectory failures and count progress."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# -I keeps the caller's environment out; this explicitly enables reference helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive51f_distribution_cases import author


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--engine", choices=("native", "reference"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "native":
        from chemworld.physchem.stochastic_adaptive_advance import (
            advance_adaptive_hybrid,
            initial_adaptive_continuation,
        )
        from chemworld.physchem.stochastic_adaptive_hybrid import MOLECULAR_QUANTUM

        network, spec, amounts = author(unit)
        (args.output / "model.json").write_text(
            json.dumps({"network": network.to_dict(), "spec": spec.to_dict()}, indent=2),
            encoding="utf-8",
        )

        def simulate(seed):
            memory = initial_adaptive_continuation(network, spec, amounts, seed=seed)
            result = advance_adaptive_hybrid(network, spec, amounts, memory, unit["duration_s"])
            return {
                "populations": {k: v / MOLECULAR_QUANTUM for k, v in result.amounts_mol.items()},
                "time_s": result.continuation["time_s"],
                "events": len(result.events),
                "partition_changes": result.partition_changes,
                "work": result.work,
                "rng_state": result.continuation["rng_state"],
            }

        base = 510600000
    else:
        from adaptive51f_distribution_reference import ReferenceKernel

        kernel = ReferenceKernel(unit)
        simulate = kernel.simulate
        base = 510700000
        assert not any(k == "chemworld" or k.startswith("chemworld.") for k in sys.modules)
    total = unit[args.engine + "_replicates"]
    started = last = time.monotonic()
    with (
        (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream,
        (args.output / "events.jsonl").open("x", encoding="utf-8") as journal,
    ):
        for replicate in range(total):
            seed = base + 100000 * unit["index"] + replicate
            identity = unit["id"] + "-" + str(replicate)
            journal.write(json.dumps({"id": identity, "event": "time_course_started"}) + "\n")
            journal.flush()
            row = {"id": identity, "seed": seed, "status": "error"}
            try:
                row.update(simulate(seed), status="ok")
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
