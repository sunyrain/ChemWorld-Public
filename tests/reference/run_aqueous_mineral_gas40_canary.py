"""One original-world integration unit, three fixed comparisons; retain all attempts."""

import argparse
import json
import traceback
from pathlib import Path

from chemworld.examples.aqueous_headspace_minerals import example_inputs
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    checks, raw = [], {}
    runtime = restored = None
    try:
        spec, initial, geometries = example_inputs()
        raw["input"] = {
            "spec": spec.to_dict(),
            "amounts": initial,
            "geometries": {m: g.to_dict() for m, g in geometries.items()},
        }
        runtime = ChemWorldRuntime(
            composition=build_aqueous_headspace_world(spec, initial, mineral_geometries=geometries)
        )
        runtime.reset(seed=40)
        original_slots = runtime._state.state_slots
        result = runtime.execute({"operation": "wait", "duration_s": 5.0})
        raw["first_result"] = result.info
        s = runtime._state
        checks.append(result.info["transaction_status"] == "committed" and s.ledger.time_s == 5.0)
        volumes = sum(p.volume_L for p in s.phases.phases.values())
        checks.append(
            s.phases.total_amounts_mol() == s.species_amounts
            and abs(volumes - s.volume_L) < 1e-12
            and s.species_amounts["H2O"] == initial["H2O"]
            and s.state_slots == original_slots
        )
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        status = []
        for world in (runtime, restored):
            status.append(
                world.execute({"operation": "wait", "duration_s": 5.0}).info["transaction_status"]
            )
        raw["final_snapshot"] = runtime.snapshot().to_dict()
        raw["restored_snapshot"] = restored.snapshot().to_dict()
        checks.append(
            status == ["committed", "committed"]
            and raw["final_snapshot"] == raw["restored_snapshot"]
        )
    except Exception:
        raw["exception"] = traceback.format_exc()
    finally:
        for world in (runtime, restored):
            if world is not None:
                world.close()
    raw["checks"] = checks
    raw["summary"] = {
        "units": 1,
        "comparisons": 3,
        "passed": sum(checks),
        "success": len(checks) == 3 and all(checks),
    }
    (output / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(json.dumps(raw["summary"]), flush=True)
    if "exception" in raw:
        print(raw["exception"], flush=True)
    return raw["summary"]["success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
