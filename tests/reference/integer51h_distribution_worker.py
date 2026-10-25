"""One fixed model/engine; independent CME or fully journaled native ensemble."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=("reference", "direct", "next", "coarse", "fine"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "reference":
        from integer51h_cme import law

        assert not any(k == "chemworld" or k.startswith("chemworld.") for k in sys.modules)
        result = law(unit)
        (args.output / "law.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {"unit": unit["id"], "reference_passed": result["passed"], **result["diagnostics"]}
            ),
            flush=True,
        )
        raise SystemExit(0 if result["passed"] else 1)
    from integer51h_distribution_cases import BASE_SEEDS, author

    from chemworld.physchem.stochastic_kinetics import advance_stochastic, initial_continuation

    network = author(unit, args.engine)
    (args.output / "model.json").write_text(
        json.dumps(network.to_dict(), indent=2), encoding="utf-8"
    )
    total = unit["replicates"]
    begun = last = time.monotonic()
    with (
        (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream,
        (args.output / "events.jsonl").open("x", encoding="utf-8") as journal,
    ):
        for replicate in range(total):
            seed = (
                17
                if unit["frozen"]
                else BASE_SEEDS[args.engine] + 100000 * unit["index"] + replicate
            )
            identity = unit["id"] + "-" + str(replicate)
            row = {
                "id": identity,
                "seed": seed,
                "status": "error",
                "simulation_attempted": False,
                "simulation_completed": False,
            }
            journal.write(json.dumps({"id": identity, "event": "trajectory_started"}) + "\n")
            journal.flush()
            try:
                memory = initial_continuation(seed, algorithm=network.stochastic["algorithm"])
                journal.write(json.dumps({"id": identity, "event": "time_course_started"}) + "\n")
                journal.flush()
                row["simulation_attempted"] = True
                result = advance_stochastic(network, unit["initial"], memory, unit["duration_s"])
                row.update(
                    status="ok",
                    simulation_completed=True,
                    counts=result.counts,
                    time_s=result.continuation["time_s"],
                    events=len(result.events),
                    batches=len(result.batches),
                    committed_firings=result.continuation["events_committed"],
                    work=result.work,
                    continuation=result.continuation,
                )
            except Exception as exc:
                row.update(error=repr(exc), traceback=traceback.format_exc())
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            now = time.monotonic()
            if now - last >= 30 or replicate + 1 == total:
                elapsed = now - begun
                print(
                    json.dumps(
                        {
                            "unit": unit["id"],
                            "engine": args.engine,
                            "completed": replicate + 1,
                            "total": total,
                            "trajectories_per_second": (replicate + 1) / elapsed
                            if elapsed
                            else None,
                            "eta_s": (total - replicate - 1) * elapsed / (replicate + 1),
                        }
                    ),
                    flush=True,
                )
                last = now


if __name__ == "__main__":
    main()
