"""Ten finite integration workflows, using the existing native session only.

Run ``python -m chemworld.examples.native_domains --output /tmp/native-domains``.
These authored synthetic examples are not autonomous agents or fitted chemistry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.coupled_system import ConstantCvThermodynamics, SpeciesInternalEnergy
from chemworld.physchem.heat_transfer_units import ThermalSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec


def mixed_clock_world():
    """Independent A/B SSA and deterministic vessels; no same-vessel hybrid."""
    from chemworld.world.biochemical_worlds import biochemical_world

    payload = biochemical_world("reversible", seed=7).to_dict()
    physics = payload["physics"]
    state = WorldState.from_dict(physics["initial_states"]["reactor"])
    physics["initial_states"]["reactor"] = state.replace(
        equipment=EquipmentLedger(
            {
                **state.equipment.equipment,
                "instrument:molecule_capture": EquipmentRecord(
                    "instrument:molecule_capture", "instrument", "reactor", batch_reset="preserve"
                ),
            }
        )
    ).to_dict()
    physics["initial_states"]["continuous"] = WorldState(
        {"A": 1e-5, "B": 0.0},
        0.01,
        350,
        101325,
        "liquid",
        "continuous",
        units=state.units,
        equipment=EquipmentLedger({}),
    ).to_dict()
    thermal = physics["models"].setdefault("thermal", {"version": "1", "bindings": {}})
    thermal["bindings"]["continuous"] = ThermalSystemSpec(
        ConstantCvThermodynamics({key: SpeciesInternalEnergy(0, 30) for key in ("A", "B")}, 20),
        pressure_law="fixed_pressure",
    ).to_dict()
    physics["lab_clock"] = LabClockSpec(analyzer="molecule_capture").to_dict()
    payload["vessels"].append({"id": "continuous", "capacity_L": 0.1})
    payload["components"] = [c for c in payload["components"] if c["kind"] != "material_input"]
    payload["components"] += [
        {
            "id": name + ":" + kind,
            "vessel": name,
            "kind": kind,
            "parameters": {"duration_range_s": [1e-6, 14400]} if kind == "lab_schedule" else {},
        }
        for name, kinds in (
            ("reactor", ("lab_schedule",)),
            ("continuous", ("thermal", "reaction", "lab_schedule")),
        )
        for kind in kinds
    ]
    payload["task"].update(
        operations=["wait", "queue_lab_measure", "select_vessel", "terminate"],
        instruments=["molecule_capture"],
    )
    return WorldCompositionSpec.from_dict(payload)


def _rename(value, old, new):
    """Rename authored entity addresses, not chemical species or model formulas."""
    if isinstance(value, dict):
        return {_rename(k, old, new): _rename(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v, old, new) for v in value]
    if isinstance(value, str) and (value == old or value.startswith(old + ":")):
        return new + value[len(old) :]
    return value


def connected_heater_world():
    """One uniquely owned heater visits surface chemistry, adsorption and SPM."""
    from chemworld.examples.native_sorbent_research import (
        synthetic_sorbent_spec,
        synthetic_sorbent_thermodynamics,
    )
    from chemworld.physchem.battery_models import BatterySPMSpec, BatteryThermalSpec
    from chemworld.world.battery_world import battery_world_spec
    from chemworld.world.sorbent_world import sorbent_world_spec
    from chemworld.world.surface_world import surface_world_spec

    gas = surface_world_spec().to_dict()
    gas["composition_id"] = "native-one-heater-three-domains"
    gas["connections"] = []
    gas["components"] = [c for c in gas["components"] if c["vessel"] != "receiver"]
    gas["vessels"] = [v for v in gas["vessels"] if v["id"] != "receiver"]
    physics = gas["physics"]
    del physics["initial_states"]["receiver"]
    del physics["models"]["thermal"]["bindings"]["receiver"]
    heater = physics["initial_states"]["reactor"]["equipment"]["reactor:heater"]
    heater["settings"]["thermal_device"]["calibration_factor"] = 1.07
    sorbent = synthetic_sorbent_spec()
    adsorption = _rename(
        sorbent_world_spec(
            sorbent,
            synthetic_sorbent_thermodynamics(sorbent),
            {"A": "C2H6O", "B": "C3H8O"},
        ).to_dict(),
        "reactor",
        "adsorber",
    )
    battery = _rename(
        battery_world_spec(
            BatterySPMSpec(thermal=BatteryThermalSpec()),
        ).to_dict(),
        "reactor",
        "cell",
    )
    for fragment, name in ((adsorption, "adsorber"), (battery, "cell")):
        local = fragment["physics"]["initial_states"][name]
        local["equipment"] = {
            key: record
            for key, record in local["equipment"].items()
            if record["equipment_type"] != "thermal_device"
        }
        physics["initial_states"][name] = local
        gas["vessels"] += [v for v in fragment["vessels"] if v["id"] == name]
        gas["components"] += [
            c
            for c in fragment["components"]
            if c["vessel"] == name and c["kind"] != "material_input"
        ]
        physics["network"]["species"] += fragment["physics"]["network"]["species"]
    model = adsorption["physics"]["models"]["thermal"]["bindings"]["adsorber"]
    physics["initial_states"]["wash_supply"] = adsorption["physics"]["initial_states"][
        "wash_supply"
    ]
    gas["vessels"] += [v for v in adsorption["vessels"] if v["id"] == "wash_supply"]
    gas["components"] += [c for c in adsorption["components"] if c["vessel"] == "wash_supply"]
    physics["models"]["thermal"]["bindings"]["adsorber"] = model
    physics["models"]["sorbent"] = adsorption["physics"]["models"]["sorbent"]
    gas["components"] += [
        {"id": name + ":heater-port", "kind": "heater_connection", "vessel": name, "parameters": {}}
        for name in ("reactor", "adsorber", "cell")
    ]
    gas["task"].update(
        budget=60,
        objective="reuse a calibrated heater without resetting chemical entities",
        operations=[
            "add_component",
            "heat",
            "wait",
            "pause_control",
            "select_vessel",
            "reconnect_heater",
            "configure_electrical_control",
            "measure",
            "terminate",
        ],
        instruments=["gas_pressure", "gas_composition", "phase_state", "battery_meter"],
    )
    return WorldCompositionSpec.from_dict(gas)


def workflow(name):
    """Fixed integration recipes, not a second capability/physics registry."""
    if name == "heater":
        spec = connected_heater_world()
        actions = [
            {"operation": "add_component", "component": "CO-feed", "amount_mol": 2e-5},
            {"operation": "add_component", "component": "O2-feed", "amount_mol": 1e-5},
            {"operation": "heat", "target_temperature_K": 355, "duration_s": 10},
            {"operation": "measure", "instrument": "gas_composition"},
            {"operation": "pause_control"},
            {"operation": "reconnect_heater", "vessel": "adsorber"},
            {"operation": "select_vessel", "vessel": "adsorber"},
            {"operation": "heat", "target_temperature_K": 355, "duration_s": 20},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "pause_control"},
            {"operation": "reconnect_heater", "vessel": "cell"},
            {"operation": "select_vessel", "vessel": "cell"},
            {"operation": "configure_electrical_control", "current_A": 0.02},
            {"operation": "heat", "target_temperature_K": 355, "duration_s": 20},
            {"operation": "measure", "instrument": "battery_meter"},
        ]
    elif name == "membrane":
        from chemworld.physchem.membranes import MembraneFoulingSpec, MembraneSpec
        from chemworld.world.membrane_world import membrane_world_spec

        spec = membrane_world_spec(MembraneSpec(fouling=MembraneFoulingSpec()))
        actions = [
            action
            for path, duration in (("process", 300), ("rinse", 60)) * 3
            for action in (
                {"operation": "set_flow_rate", "flow_rate_mL_min": 20, "flow_path": path},
                {"operation": "run_flow", "duration_s": duration},
                {"operation": "measure", "instrument": "membrane_meter"},
            )
        ]
    elif name == "mineral":
        from chemworld.world.mineral_world import mineral_world_spec

        spec = mineral_world_spec()
        actions = [
            {"operation": "wait", "duration_s": 100},
            {"operation": "add_component", "component": "HCl-solution", "amount_mol": 0.01},
            {"operation": "wait", "duration_s": 300},
            {
                "operation": "route_material",
                "connection": 0,
                "transfer_fraction": 0.25,
                "mixing": 1,
            },
            {"operation": "select_vessel", "vessel": "receiver"},
            {"operation": "sample", "sample_volume_L": 0.001},
            {"operation": "measure", "instrument": "aqueous_ph"},
        ]
    elif name == "shared-clock":
        spec = mixed_clock_world()
        actions = [
            {"operation": "wait", "duration_s": 2},
            {"operation": "queue_lab_measure"},
            {"operation": "queue_lab_measure"},
            {"operation": "wait", "duration_s": 0.25},
            {"operation": "select_vessel", "vessel": "continuous"},
            {"operation": "wait", "duration_s": 0.75},
            {"operation": "wait", "duration_s": 1},
            {"operation": "wait", "duration_s": 5},
            {"operation": "select_vessel", "vessel": "reactor"},
        ]
    elif name == "distillation":
        from chemworld.examples.native_distillation_research import (
            initial_column_state,
            synthetic_column_spec,
        )
        from chemworld.world.dynamic_column_world import dynamic_column_world_spec

        model = synthetic_column_spec()
        spec = dynamic_column_world_spec(model, initial_column_state(model))
        actions = [
            {
                "operation": "distill",
                "target_temperature_K": 430,
                "duration_s": 120,
                "reflux_ratio": 1,
            },
            {"operation": "measure", "instrument": "phase_state"},
            {
                "operation": "distill",
                "target_temperature_K": 430,
                "duration_s": 10,
                "full_reflux": 1,
                "reflux_ratio": 1,
            },
            {"operation": "collect_fraction", "transfer_fraction": 0.5},
            {
                "operation": "distill",
                "target_temperature_K": 430,
                "duration_s": 10,
                "reflux_ratio": 3,
                "full_reflux": 0,
            },
        ]
    elif name == "crystal":
        from chemworld.examples.native_crystal_research import (
            synthetic_crystal_model,
            synthetic_crystal_thermodynamics,
        )
        from chemworld.physchem.elements import molecular_weight
        from chemworld.world.crystal_world import crystal_world_spec

        spec = crystal_world_spec(
            synthetic_crystal_model(molecular_weight({"C": 6, "H": 12, "O": 1}) / 1000),
            synthetic_crystal_thermodynamics(),
            {"P": "C6H12O", "I": "C3H8O"},
        )
        actions = [
            {"operation": "seed_crystals", "seed_mass_g": 0.001},
            {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 300},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "filter_crystals"},
            {"operation": "resuspend_crystals", "volume_L": 0.01, "solvent": 0},
            {"operation": "heat", "target_temperature_K": 350, "duration_s": 60},
            {"operation": "wait", "duration_s": 60},
            {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 300},
            {"operation": "measure", "instrument": "phase_state"},
        ]
    elif name == "solid":
        from chemworld.world.solid_world import solid_world_spec

        spec = solid_world_spec()
        actions = [
            action
            for temperature in (800, 300)
            for action in (
                {"operation": "heat", "target_temperature_K": temperature, "duration_s": 120},
                {"operation": "measure", "instrument": "solid_phase_meter"},
            )
        ]
    elif name == "heterogeneous":
        from chemworld.examples.closed_shared_labs import reactive_closed_lab

        spec = reactive_closed_lab()
        actions = [
            {"operation": "select_vessel", "vessel": "adsorber"},
            {"operation": "queue_lab_heat",
             "target_temperature_K": 350, "duration_s": 2},
            {"operation": "select_vessel", "vessel": "ssa"},
            {"operation": "queue_lab_measure"},
            {"operation": "wait", "duration_s": 1},
            {"operation": "wait", "duration_s": 2},
            {"operation": "select_vessel", "vessel": "cell"},
            {"operation": "wait", "duration_s": 2},
        ]
    elif name == "open-closed":
        from chemworld.examples.open_shared_labs import open_shared_lab

        spec = open_shared_lab("transport", feed_volume_L=.001)
        actions = [
            {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
            {"operation": "select_vessel", "vessel": "bottle"},
            {"operation": "queue_lab_measure"},
            {"operation": "wait", "duration_s": 1},
            {"operation": "wait", "duration_s": 1000},
        ]
    elif name == "device-lifecycle":
        from chemworld.examples.multi_device_lab import multi_device_lab_world

        spec = multi_device_lab_world()
        actions = [
            {"operation": "queue_lab_heat", "device": "heater-a",
             "target_temperature_K": 320, "duration_s": 2},
            {"operation": "queue_lab_measure", "device": "hplc-a"},
            {"operation": "select_vessel", "vessel": "vial-c"},
            {"operation": "queue_lab_measure", "device": "hplc-a"},
            {"operation": "select_vessel", "vessel": "sample"},
            {"operation": "queue_lab_measure", "device": "hplc-b"},
            {"operation": "wait", "duration_s": 2},
            {"operation": "wait", "duration_s": 100},
            {"operation": "wait", "duration_s": 100},
            {"operation": "wait", "duration_s": 100},
            {"operation": "select_vessel", "vessel": "vial-d"},
            {"operation": "queue_lab_heat", "device": "heater-b",
             "target_temperature_K": 325, "duration_s": 2},
            {"operation": "select_vessel", "vessel": "reactor"},
            {"operation": "queue_lab_measure", "device": "hplc-a"},
            {"operation": "wait", "duration_s": 100},
            {"operation": "wait", "duration_s": 100},
        ]
    else:
        raise ValueError(f"Unknown integration workflow: {name}")
    return spec, [*actions, {"operation": "terminate"}]


WORKFLOWS = (
    "heater", "membrane", "mineral", "shared-clock", "distillation", "crystal", "solid",
    "heterogeneous", "open-closed", "device-lifecycle",
)


def workflow_initializations(name):
    """Evaluator boundaries after numbered actions, separate from Agent actions."""
    return {10: {"scope": "material", "vessel_ids": ["reactor"]}} if (
        name == "device-lifecycle"
    ) else {}


def run_workflow(name, output, *, noise_mode="sequential"):
    """Export real actions/readouts plus a restorable snapshot; no fabricated score."""
    from chemworld.data.logging import TrajectoryLogger
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.verify import verify_records

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec, actions = workflow(name)
    (output / (name + "-recipe.json")).write_text(
        json.dumps(
            {
                "composition": spec.to_dict(),
                "actions": actions,
                "initializations_after_action": workflow_initializations(name),
                "world_seed": 7,
                "observation_seed": 12,
                "noise_mode": noise_mode,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    session = ChemWorldRuntime(
        composition=spec, observation_noise_mode=noise_mode, observation_seed_override=12
    )
    session.reset(seed=7)
    path = output / (name + ".jsonl")
    try:
        with TrajectoryLogger(path) as logger:
            for index, action in enumerate(actions, 1):
                boundary = workflow_initializations(name).get(index - 1)
                if boundary is not None:
                    session.reset(options=boundary)
                result = session.execute(action)
                if result.info["transaction_status"] != "committed":
                    raise RuntimeError(
                        {
                            "workflow": name,
                            "step": index,
                            "details": {
                                key: result.info.get(key)
                                for key in (
                                    "transaction_status",
                                    "rollback_reason",
                                    "error_message",
                                    "outcome_diagnostics",
                                )
                            },
                        }
                    )
                if (
                    action["operation"] == "measure"
                    and result.info["measurement_status"] != "valid"
                ):
                    raise RuntimeError(f"{name}: invalid native readout at {index}")
                logger.log(
                    task_info={**session.task_info(), **session.evaluator_provenance()},
                    step=index,
                    action=action,
                    observation=to_observation(result.observation.values),
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                    agent_metadata={"agent_id": "fixed-native-example"},
                )
        (output / (name + "-snapshot.json")).write_text(
            json.dumps(session.snapshot().to_dict(), allow_nan=False), encoding="utf-8"
        )
        from chemworld.data.logging import load_jsonl

        replay = verify_records(load_jsonl(path), tolerance=1e-8)
        if not replay.verified:
            raise RuntimeError(replay.mismatches)
        return {
            "name": name,
            "noise_mode": noise_mode,
            "actions": len(actions),
            "replay_steps": replay.checked_steps,
            "time_s": session._state.ledger.time_s,
            "analysis_time_s": session._state.ledger.analysis_time_s,
            "sample_consumed_L": session._state.ledger.sample_consumed_L,
            "terminated": result.terminated,
            "trajectory": str(path),
        }
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow", choices=WORKFLOWS, action="append")
    parser.add_argument("--noise-mode", choices=("sequential", "keyed"), default="sequential")
    args = parser.parse_args()
    results = []
    for name in args.workflow or WORKFLOWS:
        print(f"Running native integration example: {name}", flush=True)
        results.append(run_workflow(name, args.output, noise_mode=args.noise_mode))
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
