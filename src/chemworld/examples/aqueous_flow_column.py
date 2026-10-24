"""Finite feed, reactive column and initially empty collection on original worlds."""

from dataclasses import replace
from math import sqrt
from typing import Any

from chemworld.examples.native_aqueous_reactive_transport import (
    SOURCE,
    WATER_VOLUME,
)
from chemworld.examples.native_aqueous_reactive_transport import (
    example_inputs as closed_inputs,
)
from chemworld.physchem.aqueous_reactive_transport import AqueousReactiveTransportSpec
from chemworld.physchem.aqueous_transport_flow import AqueousColumnFlowSpec


def example_inputs(
    kind: str = "slow", condition: int = 0, cells: int = 4
) -> tuple[AqueousReactiveTransportSpec, dict[str, dict[str, float]]]:
    spec, _ = closed_inputs(kind, condition, cells)
    db = spec.equilibrium.database
    # A nonzero declared water enthalpy makes missing solvent transport visible.
    water = db.solvent_species
    thermo = water.standard_state.thermo
    water = replace(
        water,
        standard_state=replace(
            water.standard_state,
            thermo=replace(thermo, reference=replace(thermo.reference, energy_J_mol=1000.0)),
        ),
    )
    db = replace(db, species=tuple(water if s.phase == "solvent" else s for s in db.species))
    flow = AqueousColumnFlowSpec(
        "feed",
        "collector",
        (2e-9, 5e-9)[condition],
        (0.0, 2e-9)[condition],
        5e-6,
        5e-6,
        5e-7,
        SOURCE,
    )
    spec = replace(spec, equilibrium=replace(spec.equilibrium, database=db), flow=flow)
    kc = (1.0, 10.0)[condition]

    def amounts(volume: float, incoming: bool) -> dict[str, float]:
        values = {"H2O": volume / WATER_VOLUME}
        if kind == "salt":
            concentration = 20.0 if incoming else 5.0
            return {**values, "M+": concentration * volume, "A-": concentration * volume}
        total = 30.0 if incoming else 10.0
        ion = 2 * kc * total / (kc + sqrt(kc * kc + 4 * kc * total))
        values.update({"H+": ion * volume, "A-": ion * volume, "HA": (total - ion) * volume})
        if kind == "slow":
            values["S"] = (8.0 if incoming else 3.0) * volume
        return values

    initial = {v: amounts(spec.vessel_capacity_m3(v), False) for v in spec.cell_vessel_ids}
    initial["feed"] = amounts(5e-6, True)
    initial["collector"] = dict.fromkeys(initial["feed"], 0.0)
    return spec, initial


def run_example() -> dict[str, Any]:
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.aqueous_reactive_transport_world import (
        build_aqueous_reactive_transport_world,
    )

    spec, amounts = example_inputs()
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, amounts))
    try:
        runtime.reset(seed=40)
        reply = runtime.execute({"operation": "wait", "duration_s": 10.0})
        return {
            "transaction": reply.info["transaction_status"],
            "time_s": runtime._state.ledger.time_s,
            "amounts": {
                v: s.species_amounts
                for v, s in {
                    runtime._state.vessel_id: runtime._state,
                    **runtime._state.inactive_vessels,
                }.items()
            },
            "bath_heat_J": runtime._state.ledger.energy_jacket_J,
        }
    finally:
        runtime.close()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), indent=2))
