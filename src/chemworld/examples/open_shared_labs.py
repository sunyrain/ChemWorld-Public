"""Finite open apparatus and an independent closed thermal bottle on one clock."""

from chemworld.examples.closed_shared_labs import _fragment
from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.lab_world import shared_lab_world


def open_shared_lab(kind="transport", *, model=None, **parameters):
    """Reuse authored finite apparatus; no new flow or membrane equations."""
    if kind == "transport":
        from chemworld.world.transport_world import transport_world_spec

        base = transport_world_spec(model, **parameters)
    elif kind == "membrane":
        from chemworld.physchem.membranes import MembraneFoulingSpec, MembraneSpec
        from chemworld.world.membrane_world import membrane_world_spec

        base = membrane_world_spec(
            model or MembraneSpec(fouling=MembraneFoulingSpec()), **parameters,
        )
    else:
        raise ValueError("choose the finite transport or membrane recipe")
    payload = base.to_dict()
    physics = payload["physics"]
    # Reuse the complete authored bottle, including the chromatograph's owned
    # retention slot. Copying only its equipment record loses real inventory.
    fragment = _fragment(shared_lab_world(), "reactor", "bottle")
    bottle = WorldState.from_dict(fragment["physics"]["initial_states"]["bottle"])
    bottle = bottle.replace(
        temperature_K=330,
        equipment=EquipmentLedger({
            key: record for key, record in bottle.equipment.equipment.items()
            if record.equipment_type != "thermal_device"
        }),
    )
    known = {species["species_id"] for species in physics["network"]["species"]}
    physics["network"]["species"] += [
        species for species in fragment["physics"]["network"]["species"]
        if species["species_id"] not in known
    ]
    physics["initial_states"]["bottle"] = bottle.to_dict()
    physics["models"]["thermal"] = fragment["physics"]["models"]["thermal"]
    payload["vessels"].append({"id": "bottle", "capacity_L": .1})
    payload["components"] += [
        {"id": "bottle:thermal", "kind": "thermal", "vessel": "bottle", "parameters": {}},
        {"id": "bottle:analyzer", "kind": "observation", "vessel": "bottle",
         "parameters": {"instruments": ["programmed_hplc"]}},
        *({"id": name + ":clock", "kind": "lab_schedule", "vessel": name,
           "parameters": {"duration_range_s": [1e-6, 14400]}}
          for name in physics["initial_states"]),
    ]
    physics["lab_clock"] = LabClockSpec(
        analyzer="programmed_hplc", vessel_capacity=len(physics["initial_states"]),
    ).to_dict()
    payload["composition_id"] = "synthetic-shared-" + kind
    payload["task"].update(
        budget=120, objective="characterize finite open and closed systems on one clock",
        operations=["set_flow_rate", "select_vessel", "queue_lab_measure", "wait", "terminate"],
        instruments=["programmed_hplc"],
    )
    return WorldCompositionSpec.from_dict(payload)
