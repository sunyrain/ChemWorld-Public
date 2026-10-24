"""Inspect finite-time perturbation growth from a native formal-world snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.examples.sbml_calibration import pool_model
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_finite_time import analyze_sbml_finite_time_world
from chemworld.world.sbml_world import sbml_world


def example_report() -> dict[str, Any]:
    composition = sbml_world(pool_model(kind="reversible", initial_b=0.2))
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=52)
        before = runtime._state.to_dict()
        options = {"sample_offsets_s": (0.2, 0.5), "amount_scales_mol": {"A": 0.8, "B": 0.2}}
        result = analyze_sbml_finite_time_world(runtime._state, composition, **options)
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        replay = analyze_sbml_finite_time_world(restored._state, composition, **options)
        return {
            "analysis": result,
            "world_unchanged": before == runtime._state.to_dict(),
            "json_replay_exact": replay == result,
        }
    finally:
        runtime.close()
        if restored:
            restored.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = example_report()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": result["analysis"]["status"],
                "world_unchanged": result["world_unchanged"],
                "json_replay_exact": result["json_replay_exact"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
