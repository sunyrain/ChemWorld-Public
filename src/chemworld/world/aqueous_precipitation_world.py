"""Declared finite solution stocks, equilibrium reactors and liquid connections."""

from collections.abc import Mapping, Sequence
from math import fsum, isfinite
from typing import Any, cast

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.aqueous_mineral_equilibrium import (
    AqueousMineralEquilibriumSpec,
    solve_aqueous_mineral_equilibrium,
)
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.aqueous_precipitation import PRECIPITATION_TYPE
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_aqueous_precipitation_world(
    spec: AqueousMineralEquilibriumSpec,
    inventories: Mapping[str, Mapping[str, float]],
    connections: Sequence[tuple[str, str, str]],
    *,
    temperature_K: float = 298.15,
    pressure_Pa: float = 1e5,
    capacities_L: Mapping[str, float] | None = None,
    allow_empty_vessels: Sequence[str] = (),
    budget: int = 100,
) -> WorldCompositionSpec:
    """Prepare equilibrium stocks; each tuple is (connection_id, source, target).

    The first named vessel is initially selected. Preparation is not experimental
    heating; route/wait subsequently settle actual Q and P dV on the common ledger.
    Mineral formation assumes instantaneous equilibrium, not nucleation kinetics.
    """
    if not inventories:
        raise ValueError("precipitation world requires finite named vessel inventories")
    if set(allow_empty_vessels) - set(inventories):
        raise ValueError("empty inventory permission requires an installed vessel")
    if len(spec.mineral_ids) != 1 or len(spec.reaction_ids) > 2:
        raise ValueError(
            "precipitation supports one pure mineral and one optional aqueous reaction"
        )
    if set(spec.dissolution_reactions) != set(spec.mineral_ids):
        raise ValueError("the pure mineral needs its declared dissolution reaction")
    capacities = dict(capacities_L or dict.fromkeys(inventories, 2.0))
    if set(capacities) != set(inventories):
        raise ValueError("each finite precipitation vessel needs its own capacity")
    if any(a not in inventories or b not in inventories or a == b for _, a, b in connections):
        raise ValueError("precipitation connections require distinct installed endpoints")
    states: dict[str, Any] = {}
    db = spec.database
    phase_ids = {k: "mineral:" + k for k in spec.mineral_ids}
    for vessel, initial in inventories.items():
        if set(initial) - {s.species_id for s in db.species} or any(
            isinstance(n, bool) or not isfinite(n) or n < 0 for n in initial.values()
        ):
            raise ValueError("initial precipitation stock needs finite declared nonnegative moles")
        empty = not any(initial.values())
        if empty and vessel in allow_empty_vessels:
            db.standard_states(temperature_K, pressure_Pa)
            amounts = dict.fromkeys((s.species_id for s in db.species), 0.0)
            initial_report = {"inventory_state": "empty", "equilibrium_performed": False}
        else:
            report = solve_aqueous_mineral_equilibrium(
                spec,
                initial,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
            )
            if not report.success or report.amounts_mol is None:
                raise ValueError(f"initial precipitation stock {vessel}: {report.failure_reason}")
            amounts = report.amounts_mol
            initial_report = report.to_dict()
        liquid = {s.species_id: amounts[s.species_id] for s in db.species if s.phase != "mineral"}
        liquid_volume_L = (
            0.0
            if empty
            else 1000
            * aqueous_phase_properties(
                db, liquid, temperature_K=temperature_K, pressure_Pa=pressure_Pa
            ).volume_m3
        )
        phases = {
            "aqueous": PhaseRecord(
                "aqueous",
                vessel,
                "aqueous",
                liquid_volume_L,
                liquid,
                selected=True,
                metadata={"carrier_representation": "species_inventory"},
            )
        }
        for species in db.species:
            if species.phase == "mineral":
                key, n = phase_ids[species.species_id], amounts[species.species_id]
                phases[key] = PhaseRecord(
                    key,
                    vessel,
                    "solid",
                    n * cast(float, species.standard_state.molar_volume_m3_mol) * 1000,
                    {species.species_id: n},
                    metadata={"pure_mineral": True},
                )
        record = EquipmentRecord(
            vessel + ":equilibrium",
            PRECIPITATION_TYPE,
            vessel,
            batch_reset="preserve",
            settings={
                "physical_model": spec.to_dict(),
                "liquid_phase_id": "aqueous",
                "mineral_phase_ids": phase_ids,
                "temperature_K": temperature_K,
                "pressure_Pa": pressure_Pa,
                **({"allow_empty_inventory": True} if vessel in allow_empty_vessels else {}),
            },
        )
        volume = fsum(p.volume_L for p in phases.values())
        if volume > capacities[vessel]:
            raise ValueError("initial precipitation inventory exceeds finite capacity")
        states[vessel] = WorldState(
            amounts,
            volume,
            temperature_K,
            pressure_Pa,
            "liquid",
            vessel,
            phases=PhaseLedger(phases),
            equipment=EquipmentLedger({record.equipment_id: record}),
            vessels=VesselLedger(
                {
                    vessel: VesselRecord(
                        vessel,
                        "declared-precipitation-TP-vessel",
                        capacities[vessel],
                        temperature_K,
                        pressure_Pa,
                        (),
                        temperature_K,
                        pressure_Pa,
                    )
                }
            ),
            metadata={"precipitation_equilibrium": initial_report},
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
    network = ReactionNetworkSpec(
        "declared-equilibrium-precipitation",
        tuple(
            SpeciesSpec(
                s.species_id,
                formula=hill_formula(dict(s.composition)),
                charge=s.charge,
                phase="solid" if s.phase == "mineral" else "liquid",
            )
            for s in db.species
        ),
        (),
    )
    physics = CompositionPhysics(network, states)
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "declared-equilibrium-precipitation",
            "components": [
                {"id": v + "-solution", "kind": "aqueous_mineral", "vessel": v, "parameters": {}}
                for v in inventories
            ],
            "vessels": [{"id": v, "capacity_L": capacities[v]} for v in inventories],
            "connections": [
                {
                    "id": key,
                    "unit": "mol",
                    "source": {"component": a + "-solution", "port": "out"},
                    "target": {"component": b + "-solution", "port": "in"},
                }
                for key, a, b in connections
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "finite precursor mixing, precipitation and dilution redissolution",
                "budget": budget,
                "operations": ["route_material", "select_vessel", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
