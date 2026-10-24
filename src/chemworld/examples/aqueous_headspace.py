"""Authored finite headspace fixtures: absorption, acid uptake and degassing."""

from math import log, sqrt
from typing import Any, Literal

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousReaction,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_gas_transfer import AqueousGasTransfer, AqueousHeadspaceSpec
from chemworld.physchem.aqueous_reaction_equilibrium import AqueousEquilibriumSpec
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo

SOURCE = "authored-aqueous-40F"
TREF = 298.15
VOLUME = 1e-4
WATER_MASS = molecular_weight(parse_formula("H2O")) / 1000


def example_inputs(
    kind: str = "acid", condition: int = 0
) -> tuple[AqueousHeadspaceSpec, dict[str, float]]:
    if kind not in {"neutral", "acid", "desorption", "double"} or condition not in (0, 1):
        raise ValueError("unknown authored headspace fixture")
    temperature = (298.15, 310.0)[condition]
    acid = kind in {"acid", "double"}
    ga = R * TREF * log(0.01) if acid else 2000.0
    rows = [
        ("H2O", "H2O", 0, "solvent", 1000.0, 0.0),
        ("I(g)", "Ar", 0, "gas", 200.0, 0.0),
        ("A(g)", "HCl", 0, "gas", 1200.0, ga + R * TREF * log(0.1)),
        ("A(aq)", "HCl", 0, "aqueous", -1000.0, ga),
    ]
    if acid:
        rows += [("H+", "H", 1, "aqueous", 0.0, 0.0), ("Cl-", "Cl", -1, "aqueous", 0.0, 0.0)]
    if kind == "double":
        rows += [
            ("B(g)", "CO2", 0, "gas", 700.0, R * TREF * log(0.2)),
            ("B(aq)", "CO2", 0, "aqueous", -500.0, 0.0),
        ]
    species = []
    for name, formula, charge, phase, h, g in rows:
        composition = parse_formula(formula)
        thermo_phase: Literal["gas", "liquid"] = "gas" if phase == "gas" else "liquid"
        cp = PropertyCorrelation(
            name + ":cp",
            "cp_polynomial",
            {"a": 10 * sum(composition.values())},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": (280.0, 330.0)},
            "Authored constant Cp; not a measured material",
            {"phase": thermo_phase},
        )
        reference = CaloricReference(
            thermo_phase,
            TREF,
            1e5,
            "enthalpy",
            h,
            (h - g) / TREF,
            "Authored shared elemental reference",
        )
        standard = AqueousStandardState(
            CorrelationSpeciesThermo(name, cp, reference, composition),
            "authored-40F-common",
            "gas-fugacity-reference" if phase == "gas" else "constant-volume",
            (1e3, 2e6),
            SOURCE,
            None if phase == "gas" else 18e-6 if phase == "solvent" else 0.0,
        )
        species.append(AqueousSpecies(name, composition, charge, phase, standard, SOURCE))
    reactions = [AqueousReaction("uptake-A", {"A(g)": -1, "A(aq)": 1}, SOURCE)]
    if acid:
        reactions.append(AqueousReaction("acid", {"A(aq)": -1, "H+": 1, "Cl-": 1}, SOURCE))
    if kind == "double":
        reactions.append(AqueousReaction("uptake-B", {"B(g)": -1, "B(aq)": 1}, SOURCE))
    db = AqueousDatabase(
        "authored-headspace-" + kind,
        (
            AqueousDataSource(
                SOURCE,
                "Authored CC0 finite gas/liquid functionality, not measured parameters",
                "urn:chemworld:authored:40F",
                "CC0-1.0",
                "https://creativecommons.org/publicdomain/zero/1.0/",
                "synthetic",
            ),
        ),
        tuple(species),
        tuple(reactions),
        ElectrolyteActivityModel(
            "authored-ideal",
            "ideal",
            tuple((s.species_id, s.charge) for s in species if s.phase == "aqueous"),
            SOURCE,
            1.0,
            (280.0, 330.0),
            (1e3, 2e6),
        ),
    )
    transfers = [AqueousGasTransfer("uptake-A", "A(g)", "A(aq)", (4e-5, 7e-5)[condition], SOURCE)]
    if kind == "double":
        transfers.append(AqueousGasTransfer("uptake-B", "B(g)", "B(aq)", 3e-5, SOURCE))
    spec = AqueousHeadspaceSpec(
        AqueousEquilibriumSpec(db, ("acid",) if acid else ()),
        tuple(transfers),
        VOLUME,
        VOLUME,
        temperature,
    )
    amounts = {s.species_id: 0.0 for s in species}
    amounts["H2O"] = VOLUME / 18e-6
    amounts["I(g)"] = 8e4 * VOLUME / (R * temperature)
    amounts["A(g)"] = (
        (0.0 if kind == "desorption" else 5e4 if acid else 2e4) * VOLUME / (R * temperature)
    )
    total = 0.002 if kind == "desorption" else 0.001 if acid else 0.0
    if acid:
        standards = db.standard_states(temperature, 1e5)
        from math import exp

        ka = exp(
            (
                standards["A(aq)"].gibbs_J_mol
                - standards["H+"].gibbs_J_mol
                - standards["Cl-"].gibbs_J_mol
            )
            / (R * temperature)
        )
        water_kg = amounts["H2O"] * WATER_MASS
        c = total / water_kg
        ion = 2 * ka * c / (ka + sqrt(ka * ka + 4 * ka * c)) * water_kg
        amounts.update({"H+": ion, "Cl-": ion, "A(aq)": total - ion})
    else:
        amounts["A(aq)"] = total
    if kind == "double":
        amounts["B(g)"] = 1e4 * VOLUME / (R * temperature)
        amounts["B(aq)"] = 0.004
    return spec, amounts


def run_example() -> dict[str, Any]:
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world

    spec, amounts = example_inputs()
    runtime = ChemWorldRuntime(composition=build_aqueous_headspace_world(spec, amounts))
    try:
        runtime.reset(seed=40)
        result = runtime.execute({"operation": "wait", "duration_s": 5.0})
        return {
            "transaction_status": result.info["transaction_status"],
            "time_s": runtime._state.ledger.time_s,
            "pressure_Pa": runtime._state.pressure_Pa,
            "amounts_mol": runtime._state.species_amounts,
            "bath_heat_J": runtime._state.ledger.energy_jacket_J,
            "diagnostics": result.info.get("outcome_diagnostics"),
        }
    finally:
        runtime.close()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), indent=2))
