"""Finite calcite lots and water receivers, through the existing world schema."""

from dataclasses import replace

from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.aqueous_mineral import CarbonateEquilibriumSpec, carbonate_feed
from chemworld.physchem.mineral_kinetics import MineralGeometry, MineralKineticsSpec
from chemworld.runtime.aqueous_mineral import aqueous_record, close_aqueous
from chemworld.runtime.mineral_kinetics import geometry_slot
from chemworld.world.aqueous_world import aqueous_world_spec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def mineral_world_spec(
    model=None, *, aqueous=None, geometry=None, amounts=None, volume_L=0.02, budget=40
):
    model, aqueous = model or MineralKineticsSpec(), aqueous or CarbonateEquilibriumSpec()
    geometry = geometry or MineralGeometry()
    if not aqueous.allow_calcite:
        raise ValueError("kinetic calcite requires its actual allowed mineral phase")
    amounts = (
        amounts
        if amounts is not None
        else carbonate_feed(
            water_kg=0.02,
            sodium_bicarbonate_mol=0.0001,
            calcium_chloride_mol=0.0001,
            sodium_chloride_mol=0.0002,
            calcite_mol=0.0001,
        )
    )
    # Prepare only the fast aqueous reactions before adding the actual seed;
    # the resulting kinetic world never performs an instantaneous mineral solve.
    base = aqueous_world_spec(
        replace(aqueous, allow_calcite=False),
        amounts={**amounts, "Calcite": 0.0},
        volume_L=volume_L,
        budget=budget,
    )
    states = dict(base.physics.initial_states)
    template = states["reactor"]

    def install(vessel, original, volume):
        record = aqueous_record(template)
        identity = f"{vessel}:carbonate-model"
        record = replace(
            record,
            equipment_id=identity,
            attached_vessel_id=vessel,
            settings={"physical_model": aqueous.to_dict(), "mineral_kinetics": model.to_dict()},
        )
        owned = (geometry_slot(vessel, geometry),) if original["Calcite"] > 0 else ()
        state = template.replace(
            vessel_id=vessel,
            thermal=None,
            volume_L=volume,
            species_amounts=original,
            species=replace(template.species, initial_amounts_mol=original),
            phases=PhaseLedger(
                {
                    "aqueous": PhaseRecord(
                        "aqueous",
                        vessel,
                        "aqueous",
                        volume,
                        {k: n for k, n in original.items() if k != "Calcite"},
                        selected=True,
                        metadata={"carrier_representation": "species_inventory"},
                    ),
                    "calcite": PhaseRecord(
                        "calcite",
                        vessel,
                        "solid",
                        0,
                        {"Calcite": original["Calcite"]},
                        metadata={"carrier_representation": "species_inventory"},
                    ),
                }
            ),
            equipment=EquipmentLedger({identity: record}),
            state_slots=StateSlots(owned),
            vessels=VesselLedger(
                {
                    vessel: VesselRecord(
                        vessel, "finite-mineral-water-vessel", 0.1, 350, 200000, (), 298.15, 101325
                    )
                }
            ),
            metadata={},
        )
        return close_aqueous(state)

    states["reactor"] = install("reactor", amounts, volume_L)
    states["receiver"] = install("receiver", carbonate_feed(water_kg=0.01), 0.01)
    physics = CompositionPhysics(
        base.physics.network, states, feed_sources=base.physics.feed_sources
    )
    payload = base.to_dict()
    payload["composition_id"] = "finite-kinetic-calcite"
    next(c for c in payload["components"] if c["id"] == "solution")["kind"] = "mineral_kinetics"
    payload["components"].extend(
        [
            {
                "id": "receiver-solution",
                "kind": "mineral_kinetics",
                "vessel": "receiver",
                "parameters": {},
            },
            {
                "id": "receiver-ph",
                "kind": "observation",
                "vessel": "receiver",
                "parameters": {"instruments": ["aqueous_ph"]},
            },
        ]
    )
    payload["vessels"] = [
        {"id": v, "capacity_L": s.vessels.vessels[v].max_volume_L} for v, s in states.items()
    ]
    payload["connections"] = [
        {
            "id": f"{a}-to-{b}-{phase}",
            "unit": "mol",
            "source": {"component": source, "port": phase},
            "target": {"component": target, "port": phase},
        }
        for a, b, source, target in (
            ("reactor", "receiver", "solution", "receiver-solution"),
            ("receiver", "reactor", "receiver-solution", "solution"),
        )
        for phase in ("calcite", "aqueous")
    ]
    payload["physics"] = physics.to_dict()
    payload["task"]["objective"] = "study finite mineral kinetics and phase-owned material transfer"
    payload["task"]["operations"] += ["select_vessel", "route_material"]
    return WorldCompositionSpec.from_dict(payload)
