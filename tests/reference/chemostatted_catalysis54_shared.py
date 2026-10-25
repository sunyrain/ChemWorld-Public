"""Separate fixed-bath shared-clock consumer; not part of the 37-unit matrix."""

import argparse
import json
import math
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from chemworld.examples.chemostatted_catalysis import example_mechanism, example_world
from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
)
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.chemostatted_catalysis import unpack_chemostatted_catalysis
from chemworld.runtime.scheduling import lab_device_choices
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics
from reference.chemostatted_catalysis54_reference import independent


def shared_world():
    world = example_world(potential=-0.2)
    spec = example_mechanism()
    pressure = 1e-6 * 8.31446261815324 * 298.15 / 1e-5
    observer = WorldState(
        {"hydrogen": 1e-6},
        0.01,
        298.15,
        pressure,
        "gas",
        "observer",
        units=dict(world.physics.initial_states["reactor"].units),
        phases=PhaseLedger(
            {"gas": PhaseRecord("gas", "observer", "gas", 0.01, {"hydrogen": 1e-6})}
        ),
        equipment=EquipmentLedger(
            {
                "observer:heater": EquipmentRecord(
                    "observer:heater",
                    "thermal_device",
                    "observer",
                    batch_reset="preserve",
                    settings={
                        "thermal_device": ThermalDeviceSpec(
                            mode="finite_power",
                            enabled=False,
                            setpoint_K=298.15,
                            minimum_setpoint_K=280,
                            maximum_setpoint_K=340,
                            maximum_power_W=1,
                            maximum_cooling_power_W=1,
                        ).to_dict()
                    },
                )
            }
        ),
    )
    thermal = world.physics.models_of("thermal")
    physics = CompositionPhysics(
        world.physics.network,
        {**world.physics.initial_states, "observer": observer},
        models={
            "surface": world.physics.models_of("surface"),
            "thermal": {
                **thermal,
                "observer": replace(
                    thermal["reactor"],
                    thermodynamics=CatalyticThermodynamics(
                        {"hydrogen": spec.species_by_id["hydrogen"]}, 0.1
                    ),
                ),
            },
        },
        lab_clock=LabClockSpec(heater_ids=("observer:heater",), vessel_capacity=2),
    )
    payload = world.to_dict()
    payload["physics"] = physics.to_dict()
    payload["vessels"].append({"id": "observer", "capacity_L": 0.1})
    payload["components"].extend(
        [
            {"id": "observer:thermal", "kind": "thermal", "vessel": "observer", "parameters": {}},
            *(
                {"id": name + ":clock", "kind": "lab_schedule", "vessel": name, "parameters": {}}
                for name in ("reactor", "observer")
            ),
        ]
    )
    payload["task"].update(
        operations=[
            "select_vessel",
            "configure_electrical_control",
            "queue_lab_heat",
            "wait",
            "terminate",
        ],
        instruments=[],
    )
    return WorldCompositionSpec.from_dict(payload)


def execute(runtime, action):
    result = runtime.execute(action)
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(
            {
                k: result.info[k]
                for k in ("transaction_status", "operation_outcome", "outcome_diagnostics")
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    target = Path(args.output)
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(json.loads(Path(args.resume).read_text()))
        execute(runtime, {"operation": "wait", "duration_s": 15})
        target.write_text(json.dumps(runtime.snapshot().to_dict(), indent=2))
        return 0
    target.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, passed, **values):
        checks.append({"name": name, "passed": bool(passed), **values})

    def number(name, actual, expected, atol, rtol=3e-7):
        check(
            name,
            math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
            actual=actual,
            expected=expected,
            atol=atol,
            rtol=rtol,
        )

    error = None
    try:
        runtime = ChemWorldRuntime(composition=shared_world())
        runtime.reset()
        execute(runtime, {"operation": "select_vessel", "vessel": "reactor"})
        initial = runtime._state
        check("fixed_bath_not_queue_choice", "heater" not in lab_device_choices(initial, "heat"))
        rejected = runtime.execute(
            {
                "operation": "queue_lab_heat",
                "device": "heater",
                "target_temperature_K": 298.15,
                "duration_s": 1,
            }
        )
        check(
            "explicit_fixed_bath_queue_rejected", rejected.info["transaction_status"] != "committed"
        )
        check(
            "rejected_queue_preserves_original_physics",
            all(
                getattr(runtime._state, key) == getattr(initial, key)
                for key in ("species_amounts", "phases", "state_slots", "equipment", "lab_clock")
            ),
        )
        number("rejected_queue_no_time", runtime._state.ledger.time_s, 0, 0, 0)
        execute(runtime, {"operation": "configure_electrical_control", "potential_V": -0.25})
        execute(runtime, {"operation": "select_vessel", "vessel": "observer"})
        execute(runtime, {"operation": "wait", "duration_s": 5})
        prefix = json.loads(json.dumps(runtime.snapshot().to_dict()))
        (target / "prefix.json").write_text(json.dumps(prefix, indent=2))
        clone = ChemWorldRuntime.from_snapshot(prefix)
        for session in (runtime, clone):
            execute(session, {"operation": "wait", "duration_s": 15})
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        (target / "snapshot.json").write_text(json.dumps(snapshot, indent=2))
        check(
            "same_process_full_JSON", json.loads(json.dumps(clone.snapshot().to_dict())) == snapshot
        )
        child = subprocess.run(
            [
                sys.executable,
                __file__,
                "--resume",
                str(target / "prefix.json"),
                "--output",
                str(target / "continued.json"),
            ],
            capture_output=True,
            text=True,
            timeout=50,
        )
        (target / "child.json").write_text(
            json.dumps(
                {"returncode": child.returncode, "stdout": child.stdout, "stderr": child.stderr},
                indent=2,
            )
        )
        check(
            "fresh_process_full_JSON",
            child.returncode == 0
            and json.loads((target / "continued.json").read_text()) == snapshot,
        )
        state = runtime._state
        cell = state.inactive_vessels["reactor"]
        _, _, _, slot, physical, _, _ = unpack_chemostatted_catalysis(cell)
        expected = independent("her", 298.15, 2, -0.25, 20)
        check("external_vessel_remains_selected", state.vessel_id == "observer")
        number("global_time_s", state.ledger.time_s, 20, 1e-12, 0)
        for name in ("reactor", "observer"):
            number(name + "_elapsed_s", state.vessel_elapsed_s[name], 20, 1e-12, 0)
        check(
            "real_liquid_unchanged",
            cell.phases.phases["electrolyte"] == initial.phases.phases["electrolyte"],
        )
        number("observer_stock_mol", state.species_amounts["hydrogen"], 1e-6, 0, 0)
        number("coverage", slot.values["H_ads"] / 1e-6, expected["theta"], 3e-10)
        number("gas_mol", physical.amounts_mol["hydrogen"], expected["gas_mol"], 3e-15)
        for name in ("faradaic_charge_C", "electrical_work_J"):
            number(name, physical.coordinates[name], expected[name], 1e-10)
        number(
            "global_work_once",
            state.ledger.flow_electrical_work_J,
            expected["electrical_work_J"],
            1e-10,
        )
        number("global_bath_heat_J", state.ledger.energy_jacket_J, expected["bath_heat_J"], 1e-9)
        number(
            "cell_local_heat_J",
            cell.thermal.vessels["reactor"].energy_jacket_J,
            expected["bath_heat_J"],
            1e-9,
        )
        number("observer_local_heat_J", state.thermal.vessels["observer"].energy_jacket_J, 0, 1e-12)
    except Exception:
        error = traceback.format_exc()
    result = {
        "mode": "separate-development-shared-clock-consumer",
        "units": 1,
        "passed_units": int(error is None and all(c["passed"] for c in checks)),
        "checks": checks,
        "total_checks": len(checks),
        "passed_checks": sum(c["passed"] for c in checks),
        "error": error,
    }
    (target / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)
    return 0 if result["passed_units"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
