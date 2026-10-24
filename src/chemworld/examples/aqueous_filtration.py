"""Recover and wash a finite wet cake with explicitly synthetic aqueous chemistry.

Run: uv run --no-sync python -m chemworld.examples.aqueous_filtration
No measured salt accuracy, filter kinetics, pressure drop or drying is claimed.
"""

from chemworld.examples.aqueous_precipitation import WATER_MASS, route, synthetic_specification
from chemworld.physchem.aqueous_mineral_equilibrium import AqueousMineralEquilibriumSpec
from chemworld.runtime import ChemWorldRuntime, SessionResult
from chemworld.world.aqueous_filtration_world import (
    AqueousFilterSpec,
    build_aqueous_filtration_world,
)
from chemworld.world.composition import WorldCompositionSpec


def synthetic_filtration_world(
    family: str = "Davies",
    *,
    solid_retention: float = 0.9,
    liquid_retention: float = 0.08,
    association: bool = False,
) -> tuple[AqueousMineralEquilibriumSpec, WorldCompositionSpec]:
    spec = synthetic_specification(family, association=association)
    blank = dict.fromkeys((s.species_id for s in spec.database.species), 0.0)
    inventories = {
        "feed-a": {**blank, "H2O": 0.5 / WATER_MASS, "Na+": 0.1, "Br-": 0.1},
        "feed-b": {**blank, "H2O": 0.5 / WATER_MASS, "K+": 0.1, "Cl-": 0.1},
        "reactor": {**blank, "H2O": 0.05 / WATER_MASS},
        "diluent": {**blank, "H2O": 8 / WATER_MASS},
        "cake": dict(blank),
        "filtrate": dict(blank),
        "washed-cake": dict(blank),
        "wash-filtrate": dict(blank),
    }
    filters = (
        AqueousFilterSpec(
            "first-filter", "reactor", "cake", "filtrate", solid_retention, liquid_retention
        ),
        AqueousFilterSpec(
            "wash-filter", "cake", "washed-cake", "wash-filtrate", solid_retention, liquid_retention
        ),
    )
    connections = (
        ("feed-a-reactor", "feed-a", "reactor"),
        ("feed-b-reactor", "feed-b", "reactor"),
        ("wash-cake", "diluent", "cake"),
    )
    return spec, build_aqueous_filtration_world(spec, inventories, filters, connections)


def filter_vessel(session: ChemWorldRuntime, vessel: str) -> SessionResult:
    if session._state.vessel_id != vessel:
        index = session._state.metadata["component_network"]["vessels"].index(vessel)
        selected = session.execute({"operation": "select_vessel", "vessel": index})
        if selected.info["transaction_status"] != "committed":
            raise RuntimeError(selected.info)
    result = session.execute({"operation": "filter_crystals"})
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
    return result


def main() -> None:
    _, composition = synthetic_filtration_world()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=41)
    try:
        route(session, "feed-a", "feed-a-reactor", 0.5)
        route(session, "feed-b", "feed-b-reactor", 0.5)
        filter_vessel(session, "reactor")
        print("First source empty:", not any(session._state.species_amounts.values()))
        route(session, "diluent", "wash-cake", 0.025)
        filter_vessel(session, "cake")
        for key in ("filtrate", "washed-cake", "wash-filtrate"):
            state = session._state.inactive_vessels[key]
            print(key, "solid mol:", state.species_amounts["NaCl(s)"], "volume L:", state.volume_L)
        print("Physical time s:", session._state.ledger.time_s)
    finally:
        session.close()


if __name__ == "__main__":
    main()
