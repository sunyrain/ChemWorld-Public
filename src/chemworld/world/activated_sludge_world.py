"""The complete empirical ASM1 process family on original finite ChemWorld tanks."""

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.activated_sludge import ActivatedSludgeInventory, validate_inventory
from chemworld.physchem.activated_sludge_specs import ASM1_EQUIPMENT, ActivatedSludgeSpec
from chemworld.runtime.activated_sludge import install_activated_sludge_tank
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def activated_sludge_world(
    spec: ActivatedSludgeSpec,
    initial: ActivatedSludgeInventory,
    *,
    shared_clock: bool = True,
    budget: int = 100,
) -> WorldCompositionSpec:
    validate_inventory(spec, initial)
    if (
        initial.time_s != 0
        or any(v for row in initial.extents for v in row.values())
        or any(initial.work.values())
    ):
        raise ValueError(
            "ASM1 authoring requires a fresh process, not editable past reaction history"
        )
    names, states = spec.member_vessel_ids, {}
    for i, (name, capacity) in enumerate(zip(names, spec.capacities_m3, strict=True)):
        records = {}
        if i == 0:
            records["asm1_treatment"] = EquipmentRecord(
                "asm1_treatment",
                ASM1_EQUIPMENT,
                name,
                batch_reset="preserve",
                settings={
                    "physical_model": spec.to_dict(),
                    "empirical_reaction_balance": spec.balance_declaration(),
                },
            )
        elif name in spec.reactor_vessel_ids:
            records[f"asm1:{name}"] = EquipmentRecord(
                f"asm1:{name}", "asm1_bioreactor", name, batch_reset="preserve"
            )
        state = WorldState(
            {"H2O": initial.water_mol[i]},
            0.0,
            spec.temperature_K,
            spec.pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(records),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-asm1-tank",
                        capacity * 1000,
                        323.15,
                        spec.pressure_Pa * 1.1,
                        ("liquid",),
                        spec.temperature_K,
                        spec.pressure_Pa,
                    )
                }
            ),
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
                "cost": "currency",
                "risk": "risk",
            },
        )
        states[name] = install_activated_sludge_tank(state, spec, i, initial)
    components = [
        {"id": names[0], "kind": "finite_transport", "vessel": names[0], "parameters": {}}
    ]
    components.extend(
        {"id": n, "kind": "material_input", "vessel": n, "parameters": {}} for n in names[1:]
    )
    if shared_clock:
        components.extend(
            {
                "id": f"{n}:clock",
                "kind": "lab_schedule",
                "vessel": n,
                "parameters": {"duration_range_s": [1e-6, 14400.0]},
            }
            for n in names
        )
    pairs = [
        (names[0], names[1]),
        *((names[i], names[i + 1]) for i in range(1, len(spec.reactor_vessel_ids))),
        (names[-3], names[-2]),
        (names[-3], names[-1]),
    ]
    if len(spec.reactor_vessel_ids) > 1:
        pairs.append((names[-3], names[1]))
    physics = CompositionPhysics(
        spec.network,
        states,
        lab_clock=LabClockSpec(vessel_capacity=len(names)) if shared_clock else None,
        advance_bindings={names[0]: "finite-pump-transport"},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": f"asm1-{len(names) - 3}-cstr",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": c * 1000}
                for n, c in zip(names, spec.capacities_m3, strict=True)
            ],
            "connections": [
                {
                    "id": f"asm1-flow:{a}:{b}",
                    "unit": "mol",
                    "source": {"component": a, "port": "out"},
                    "target": {"component": b, "port": "in"},
                }
                for a, b in pairs
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study a complete ASM1 finite wastewater workflow",
                "budget": budget,
                "operations": [
                    "set_flow_rate",
                    "wait",
                    "select_vessel",
                    "terminate",
                    *([] if shared_clock else ["run_flow"]),
                ],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
