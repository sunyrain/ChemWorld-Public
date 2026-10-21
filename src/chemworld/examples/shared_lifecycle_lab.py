"""Finite sorbent loading, material transfer and real wash on a shared clock."""

from chemworld.examples.native_sorbent_research import (
    synthetic_sorbent_spec,
    synthetic_sorbent_thermodynamics,
)
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.sorbent_world import sorbent_world_spec


def sample_archive_lab():
    from chemworld.examples.multi_device_lab import multi_device_lab_world

    payload = multi_device_lab_world().to_dict()
    payload["composition_id"] = "multi-device-with-explicit-sample-storage"
    # Storage keeps the existing sealed, prequenched specimen contract. The
    # authored reporter mixture has no reaction or new quenching operation.
    payload["physics"]["initial_states"]["reactor"]["quenched"] = True
    payload["components"].append({
        "id": "reactor:sample-storage", "kind": "sample_storage", "vessel": "reactor",
        "parameters": {},
    })
    payload["task"]["operations"] += ["create_container", "transfer_material"]
    return WorldCompositionSpec.from_dict(payload)


def sorbent_maintenance_lab():
    spec = synthetic_sorbent_spec()
    payload = sorbent_world_spec(
        spec, synthetic_sorbent_thermodynamics(spec), {"A": "C2H6O", "B": "C3H8O"},
    ).to_dict()
    payload["composition_id"] = "finite-shared-sorbent-maintenance"
    physics = payload["physics"]
    for edge in payload["connections"]:
        # These are explicit zero-time aliquot moves, never continuously open
        # paths or extra owners of a coupled advance system.
        edge["scheduling"] = "manual"
    payload["components"] = [c for c in payload["components"] if c["kind"] != "observation"]
    for name, state in physics["initial_states"].items():
        if name != "reactor":
            state["equipment"] = {
                identity: record for identity, record in state["equipment"].items()
                if record["equipment_type"] != "thermal_device"
            }
        payload["components"].append({
            "id": name + ":shared-clock", "kind": "lab_schedule", "vessel": name,
            "parameters": {"duration_range_s": [1e-6, 14400]},
        })
    physics["lab_clock"] = LabClockSpec(heater_id="reactor:heater", vessel_capacity=4).to_dict()
    payload["task"].update(
        budget=160, instruments=[],
        operations=["wait", "queue_lab_heat", "add_solvent", "route_material", "select_vessel",
                    "terminate"],
    )
    return WorldCompositionSpec.from_dict(payload)
