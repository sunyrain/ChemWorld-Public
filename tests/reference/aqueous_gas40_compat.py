"""Real v1 consumer runnable against either pre-40G or current source."""

import argparse
import json
from pathlib import Path

from chemworld.examples.aqueous_headspace import example_inputs
from chemworld.physchem.aqueous_gas_transfer import advance_aqueous_headspace
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world


def run(kind, condition, legacy=None):
    spec, amounts = example_inputs(kind, condition)
    direct = advance_aqueous_headspace(spec, amounts, 30.0, evaluation_times_s=(1.0, 5.0))
    if legacy is not None:
        snapshot = json.loads(legacy.read_text())["world"]["snapshot"]
        worlds = [ChemWorldRuntime.from_snapshot(snapshot) for _ in range(2)]
        try:
            results = [w.execute({"operation": "wait", "duration_s": 5.0}).info for w in worlds]
            return {
                "direct": {"spec": spec.to_dict(), "result": direct.to_dict()},
                "world": {
                    "snapshot": snapshot,
                    "results": results,
                    "final": worlds[0].snapshot().to_dict(),
                    "restored": worlds[1].snapshot().to_dict(),
                },
            }
        finally:
            for world in worlds:
                world.close()
    composition = build_aqueous_headspace_world(spec, amounts)
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=40)
        first = runtime.execute({"operation": "wait", "duration_s": 5.0})
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        results = [
            r.execute({"operation": "wait", "duration_s": 5.0}).info for r in (runtime, restored)
        ]
        return {
            "direct": {"spec": spec.to_dict(), "result": direct.to_dict()},
            "world": {
                "composition": composition.to_dict(),
                "first_info": first.info,
                "snapshot": snapshot,
                "results": results,
                "final": runtime.snapshot().to_dict(),
                "restored": restored.snapshot().to_dict(),
            },
        }
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(run(args.kind, args.condition, args.legacy_source), indent=2), encoding="utf-8"
    )
