"""Retain every step of the three fixed IF97 native equipment histories."""

import argparse
import json
import math
from pathlib import Path

from tests.test_water_if97_world import inputs, packed

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot

ACTIONS = (
    ("heat", {"target_temperature_K": 600, "duration_s": 2}, 2, 20),
    ("wait", {"duration_s": 2}, 2, 20), ("pause_control", {}, 0, 0),
    ("wait", {"duration_s": 1}, 1, 0), ("resume_control", {}, 0, 0),
    ("wait", {"duration_s": 2}, 2, 20),
)


def run():
    cases = []
    for kind in ("liquid", "vapor", "coexistence"):
        runtime = restored = None
        case = {"kind": kind, "steps": [], "passed": False}
        try:
            payload, model = inputs(kind)
            thermal = model.thermodynamics
            runtime = ChemWorldRuntime(composition=payload)
            runtime.reset(seed=25)
            initial = runtime._state
            initial_energy = thermal.internal_energy_j(packed(initial))
            case["initial"] = {"temperature_K": initial.temperature_K,
                               "pressure_Pa": initial.pressure_Pa, "volume_L": initial.volume_L,
                               "amounts_mol": dict(initial.species_amounts),
                               "internal_energy_J": initial_energy}
            seconds = heat = 0
            for operation, parameters, duration, energy in ACTIONS:
                action = {"operation": operation, **parameters}
                result = runtime.execute(action)
                seconds += duration
                heat += energy
                if restored is not None:
                    restored.execute(action)
                state = runtime._state
                actual_energy = thermal.internal_energy_j(packed(state)) - initial_energy
                current = thermal.water_state(packed(state))
                checks = {
                    "committed": result.info["transaction_status"] == "committed",
                    "inventory": state.species_amounts == initial.species_amounts,
                    "volume": state.volume_L == initial.volume_L,
                    "clock": state.ledger.time_s == seconds,
                    "heat_ledger": math.isclose(state.ledger.energy_jacket_J, heat,
                                                rel_tol=0, abs_tol=1e-4),
                    "internal_energy": math.isclose(actual_energy, heat, rel_tol=0, abs_tol=1e-4),
                    "pressure": math.isclose(state.pressure_Pa, current.pressure_Pa, rel_tol=2e-8),
                    "snapshot": restored is None or restored._state.to_dict() == state.to_dict(),
                }
                case["steps"].append({"action": action, "checks": checks,
                    "temperature_K": state.temperature_K, "pressure_Pa": state.pressure_Pa,
                    "vapor_mass_fraction": current.vapor_mass_fraction,
                    "time_s": state.ledger.time_s, "heat_J": state.ledger.energy_jacket_J,
                    "internal_energy_change_J": actual_energy,
                    "energy_residual_J": actual_energy - heat,
                    "runtime_diagnostics": result.info.get("outcome_diagnostics", {})})
                if restored is None:
                    snapshot = SessionSnapshot.from_dict(
                        json.loads(json.dumps(runtime.snapshot().to_dict())))
                    restored = ChemWorldRuntime.from_snapshot(snapshot)
            case["passed"] = all(all(step["checks"].values()) for step in case["steps"])
        except Exception as exc:
            case["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if runtime is not None:
                runtime.close()
            if restored is not None:
                restored.close()
        cases.append(case)
        print(f"water history {len(cases)}/3: {kind}; passed={case['passed']}", flush=True)
    return {"development_evidence_only": True, "histories": 3,
            "planned_steps": 18, "completed_steps": sum(len(c["steps"]) for c in cases),
            "passed": sum(c["passed"] for c in cases),
            "failed": sum(not c["passed"] for c in cases), "cases": cases,
            "scope": "Original closed-water runtime and equipment; "
                     "no open flow or wet-lab validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("retain prior runs; choose a fresh path")
    result = run()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))
    raise SystemExit(bool(result["failed"]))
