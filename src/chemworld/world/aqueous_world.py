"""A finite carbonate vessel and actual neutral acid/base solution stocks."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.aqueous_mineral import (
    CHARGES,
    FORMULAS,
    CarbonateEquilibriumSpec,
    carbonate_feed,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.aqueous_mineral import AQUEOUS_TYPE, close_aqueous
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def aqueous_world_spec(model=None, *, amounts=None, volume_L=0.02, budget=40):
    model = model or CarbonateEquilibriumSpec()
    amounts = (
        amounts
        if amounts is not None
        else carbonate_feed(
            water_kg=0.02,
            sodium_bicarbonate_mol=0.0001,
            calcium_chloride_mol=0.0001,
            sodium_chloride_mol=0.0002,
        )
    )
    units = {
        "amount": "mol",
        "volume": "L",
        "temperature": "K",
        "pressure": "Pa",
        "time": "s",
        "cost": "currency",
        "risk": "risk",
    }

    def local(vessel, feed, volume, capacity, *, reactive=False):
        records = (
            {}
            if not reactive
            else {
                "carbonate-model": EquipmentRecord(
                    "carbonate-model",
                    AQUEOUS_TYPE,
                    vessel,
                    batch_reset="preserve",
                    settings={"physical_model": model.to_dict()},
                )
            }
        )
        state = WorldState(
            feed,
            volume,
            298.15,
            101325,
            "liquid",
            vessel,
            units=units,
            equipment=EquipmentLedger(records),
            vessels=VesselLedger(
                {
                    vessel: VesselRecord(
                        vessel, "finite-aqueous-vessel", capacity, 350, 200000, (), 298.15, 101325
                    )
                }
            ),
            phases=PhaseLedger(
                {
                    "aqueous": PhaseRecord(
                        "aqueous",
                        vessel,
                        "aqueous",
                        volume,
                        feed,
                        selected=True,
                        metadata={"carrier_representation": "species_inventory"},
                    )
                }
            ),
        )
        return close_aqueous(state) if reactive else state

    states = {
        "reactor": local("reactor", amounts, volume_L, 0.1, reactive=True),
        "acid_stock": local(
            "acid_stock", carbonate_feed(water_kg=0.005, hydrochloric_acid_mol=0.0002), 0.005, 0.01
        ),
        "base_stock": local(
            "base_stock", carbonate_feed(water_kg=0.005, sodium_hydroxide_mol=0.0002), 0.005, 0.01
        ),
    }
    network = ReactionNetworkSpec(
        "bounded-carbonate-calcite",
        tuple(
            SpeciesSpec(k, f, phase="solid" if k == "Calcite" else "liquid", charge=CHARGES[k])
            for k, f in FORMULAS.items()
        ),
        (),
    )
    physics = CompositionPhysics(
        network, states, feed_sources={"HCl-solution": "acid_stock", "NaOH-solution": "base_stock"}
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-carbonate-calcite",
            "components": [
                {
                    "id": "solution",
                    "kind": "aqueous_mineral",
                    "vessel": "reactor",
                    "parameters": {},
                },
                {
                    "id": "ph",
                    "kind": "observation",
                    "vessel": "reactor",
                    "parameters": {"instruments": ["aqueous_ph"]},
                },
                *(
                    {"id": name, "kind": "material_input", "vessel": name, "parameters": {}}
                    for name in ("acid_stock", "base_stock")
                ),
            ],
            "vessels": [
                {"id": k, "capacity_L": s.vessels.vessels[k].max_volume_L}
                for k, s in states.items()
            ],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite carbonate preparation and mineral equilibrium",
                "budget": budget,
                "operations": ["add_component", "wait", "sample", "measure", "terminate"],
                "instruments": ["aqueous_ph"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
