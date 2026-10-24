"""Finite mixing/precipitation/dilution with explicitly synthetic thermochemistry.

Run: uv run --no-sync python -m chemworld.examples.aqueous_precipitation
Chemical labels identify balanced species; these are NOT measured salt parameters.
"""

from dataclasses import replace
from math import log
from typing import Literal

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousReaction,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_mineral_equilibrium import AqueousMineralEquilibriumSpec
from chemworld.physchem.electrolyte_activity import (
    ElectrolyteActivityModel,
    ElectrolyteInteraction,
    ElectrolyteParameter,
    electrolyte_activity_catalog,
)
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo
from chemworld.runtime import ChemWorldRuntime, SessionResult
from chemworld.world.aqueous_precipitation_world import build_aqueous_precipitation_world
from chemworld.world.composition import WorldCompositionSpec

SOURCE = "synthetic-precipitation40H"
WATER_MASS = molecular_weight(parse_formula("H2O")) / 1000
R = 8.31446261815324


def synthetic_specification(
    family: str = "Davies", *, association: bool = False
) -> AqueousMineralEquilibriumSpec:
    """Author one shared H/G/V reference and complete declared activity parameters."""
    rows = [
        ("H2O", "H2O", 0, "solvent", 0.0, 0.0),
        ("Na+", "Na", 1, "aqueous", 0.0, 0.0),
        ("Cl-", "Cl", -1, "aqueous", 0.0, 0.0),
        ("K+", "K", 1, "aqueous", 0.0, 0.0),
        ("Br-", "Br", -1, "aqueous", 0.0, 0.0),
        ("NaCl(s)", "NaCl", 0, "mineral", -1000.0, -1000 / 298.15 - R * log(1e-4)),
    ]
    if association:
        rows.append(("NaCl(aq)", "NaCl", 0, "aqueous", -300.0, -300 / 298.15 + R * log(10)))
    species = []
    for key, formula, charge, phase, h, s in rows:
        composition = parse_formula(formula)
        atoms = sum(composition.values())
        caloric_phase: Literal["solid", "liquid"] = "solid" if phase == "mineral" else "liquid"
        cp = PropertyCorrelation(
            key + "-cp",
            "cp_polynomial",
            {"a": 10.0 * atoms},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": (275.0, 350.0)},
            "Synthetic elemental-additive Cp",
            {"phase": caloric_phase},
        )
        reference = CaloricReference(
            caloric_phase, 298.15, 1e5, "enthalpy", h, s, "Synthetic common elemental gauge"
        )
        thermo = CorrelationSpeciesThermo(key, cp, reference, composition)
        standard = AqueousStandardState(
            thermo,
            SOURCE,
            "constant-volume",
            (1e5, 1e6),
            SOURCE,
            atoms * 1e-6 * (0.9 if phase == "mineral" else 1.0),
        )
        species.append(AqueousSpecies(key, composition, charge, phase, standard, SOURCE))
    catalog = electrolyte_activity_catalog()[family]
    values = {
        "A": 0.509,
        "B": 1.5,
        "Aphi": 0.392,
        "b": 1.2 if family == "Pitzer" else 0.3,
        "epsilon": 0.03,
        "beta0": 0.07,
        "beta1": 0.2,
        "beta2": 0.015,
        "C0": 0.0004,
        "C1": 0.0002,
        "alpha1": 2.0,
        "alpha2": 12.0,
        "omega": 2.5,
        "theta": 0.03,
        "psi": 0.005,
        "lambda": 0.02,
        "zeta": 0.007,
        "mu": 0.001,
    }

    def parameter(key: str, unit: str) -> ElectrolyteParameter:
        return ElectrolyteParameter(
            values[key],
            unit,
            SOURCE,
            (275.0, 350.0),
            (1e5, 1e6),
            per_K=values[key] * 0.001 if key in {"A", "Aphi", "epsilon", "beta0"} else 0.0,
            per_Pa=values[key] * 1e-8 if key in {"A", "Aphi", "epsilon", "beta0"} else 0.0,
        )

    activities = ElectrolyteActivityModel(
        SOURCE + family,
        family,
        tuple((s.species_id, s.charge) for s in species if s.phase == "aqueous"),
        SOURCE,
        1.0,
        (275.0, 350.0),
        (1e5, 1e6),
        {k: parameter(k, unit) for k, unit in catalog["parameters"].items()},
    )
    activities = replace(
        activities,
        interactions=tuple(
            ElectrolyteInteraction(
                kind,
                ids,
                {k: parameter(k, unit) for k, unit in catalog["interactions"][kind].items()},
            )
            for kind, ids in activities.required_interactions()
        ),
    )
    reactions = [AqueousReaction("dissolve", {"NaCl(s)": -1, "Na+": 1, "Cl-": 1}, SOURCE)]
    if association:
        reactions.append(
            AqueousReaction("associate", {"Na+": -1, "Cl-": -1, "NaCl(aq)": 1}, SOURCE)
        )
    source = AqueousDataSource(
        SOURCE,
        "Artificial metathesis parameters; no material accuracy",
        "urn:chemworld:synthetic:40H",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
    )
    return AqueousMineralEquilibriumSpec(
        AqueousDatabase(SOURCE + family, (source,), tuple(species), tuple(reactions), activities),
        tuple(r.reaction_id for r in reactions),
    )


def synthetic_world(
    family: str = "Davies", *, association: bool = False
) -> tuple[AqueousMineralEquilibriumSpec, WorldCompositionSpec]:
    spec = synthetic_specification(family, association=association)
    blank = dict.fromkeys((s.species_id for s in spec.database.species), 0.0)
    inventories = {
        "feed-a": {**blank, "H2O": 0.5 / WATER_MASS, "Na+": 0.1, "Br-": 0.1},
        "feed-b": {**blank, "H2O": 0.5 / WATER_MASS, "K+": 0.1, "Cl-": 0.1},
        "reactor": {**blank, "H2O": 0.05 / WATER_MASS},
        "diluent": {**blank, "H2O": 8 / WATER_MASS},
    }
    connections = [
        ("feed-a-reactor", "feed-a", "reactor"),
        ("feed-b-reactor", "feed-b", "reactor"),
        ("dilute-reactor", "diluent", "reactor"),
        ("withdraw-reactor", "reactor", "feed-a"),
    ]
    return spec, build_aqueous_precipitation_world(spec, inventories, connections)


def route(
    session: ChemWorldRuntime, source: str, connection: str, fraction: float
) -> SessionResult:
    network = session._state.metadata["component_network"]
    if session._state.vessel_id != source:
        result = session.execute(
            {"operation": "select_vessel", "vessel": network["vessels"].index(source)}
        )
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
    result = session.execute(
        {
            "operation": "route_material",
            "connection": next(
                i for i, e in enumerate(network["connections"]) if e["id"] == connection
            ),
            "transfer_fraction": fraction,
            "mixing": 1,
        }
    )
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
    return result


def main() -> None:
    _, composition = synthetic_world()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=40)
    try:
        for source, edge, fraction in (
            ("feed-a", "feed-a-reactor", 0.5),
            ("feed-b", "feed-b-reactor", 0.5),
            ("diluent", "dilute-reactor", 0.25),
            ("diluent", "dilute-reactor", 0.75),
        ):
            route(session, source, edge, fraction)
            reactor = session._state.inactive_vessels["reactor"]
            print(
                edge,
                "mineral mol:",
                reactor.species_amounts["NaCl(s)"],
                "net heat J:",
                session._state.ledger.energy_jacket_J - session._state.ledger.heat_loss_J,
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
