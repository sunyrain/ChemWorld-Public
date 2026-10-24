"""Fixed small 42D development canary; retain complete outputs outside Git."""

import argparse
import json
import time
from pathlib import Path
from threading import Event, Thread

from chemworld.examples.calphad_ternary import build_calphad_ternary_example, ternary_example_spec
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.calphad_ternary_diagram import (
    CalphadTernaryResult,
    scan_calphad_ternary_diagram,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget
from chemworld.runtime.calphad_ternary_inventory import select_calphad_ternary_inventory_point
from chemworld.world.calphad_ternary_analysis import analyze_calphad_ternary_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"stage": "canary", "maps": [], "world": {}, "failures": []}
    start = time.monotonic()
    done = Event()

    def heartbeat():
        while not done.wait(30):
            print(
                f"canary {summary['stage']}: maps={len(summary['maps'])}/3 "
                f"elapsed={time.monotonic() - start:.1f}s",
                flush=True,
            )

    Thread(target=heartbeat, daemon=True).start()
    for family in ("preferred", "ideal", "regular"):
        result = scan_calphad_ternary_diagram(
            ternary_example_spec(family),
            progress=lambda p, family=family: print(family, json.dumps(p), flush=True),
        )
        encoded = result.to_dict()
        (args.output / f"{family}.json").write_text(json.dumps(encoded, indent=2), encoding="utf8")
        CalphadTernaryResult.from_dict(json.loads(json.dumps(encoded, allow_nan=False)))
        failures = [
            {"family": family, "point": p["index"], "reason": p["equilibrium"]["failure_reason"]}
            for p in result.points
            if not p["success"]
        ]
        summary["maps"].append(
            {
                "family": family,
                "points": len(result.points),
                "passed": len(result.points) - len(failures),
            }
        )
        summary["failures"].extend(failures)
        if family == "preferred":
            ledger = PhaseLedger(
                {
                    "feed": PhaseRecord(
                        "feed",
                        "reactor",
                        "solid",
                        0.00002,
                        {"FE": 0.0006, "NI": 0.0006, "CR": 0.0008},
                    )
                }
            )
            selected = select_calphad_ternary_inventory_point(
                ledger,
                result,
                10,
                vessel_id="reactor",
                source_phase_ids=("feed",),
                phase_targets=tuple(
                    CalphadPhaseTarget(m.phase_name.lower(), m.phase_name, 0.01)
                    for m in result.specification.equilibrium.models
                ),
            )
            summary["inventory"] = selected.to_dict()
    print("canary original thermal world construction", flush=True)
    summary["stage"] = "original thermal world"
    session = replay = None
    try:
        composition = build_calphad_ternary_example()
        session = ChemWorldRuntime(composition=composition)
        session.reset(seed=42)
        before = session._state.to_dict()
        report = analyze_calphad_ternary_world(
            composition,
            session._state,
            grid_intervals=3,
            progress=lambda p: print("world", p, flush=True),
        )
        summary["world"]["query"] = report
        summary["world"]["read_only"] = before == session._state.to_dict()
        action = session.execute(
            {"operation": "heat", "target_temperature_K": 800.0, "duration_s": 1.0}
        )
        summary["world"]["heat"] = action.info
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        for runtime in (session, replay):
            result = runtime.execute({"operation": "wait", "duration_s": 1.0})
            if result.info["transaction_status"] != "committed":
                summary["failures"].append({"world_wait": result.info})
        summary["world"]["replay_equal"] = replay._state.to_dict() == session._state.to_dict()
        summary["world"]["state"] = session._state.to_dict()
    except Exception as exc:
        summary["failures"].append({"world": repr(exc)})
    finally:
        if session is not None:
            session.close()
        if replay is not None:
            replay.close()
    summary["elapsed_s"] = time.monotonic() - start
    done.set()
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf8"
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"world", "inventory"}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
