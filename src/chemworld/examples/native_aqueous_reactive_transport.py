"""Authored CC0 spatial electrolyte fixtures; no measured material parameters."""

from math import cos, log, pi, sin, sqrt
from typing import Any

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousReaction,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_electrodiffusion import AqueousDiffusionCoefficient
from chemworld.physchem.aqueous_reaction_equilibrium import AqueousEquilibriumSpec
from chemworld.physchem.aqueous_reactive_transport import (
    AqueousReactiveTransportSpec,
    AqueousSlowConversion,
)
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo

SOURCE = "authored-aqueous-40D"
TEMPERATURE = 298.15
WATER_VOLUME = 18e-6
WATER_MASS = molecular_weight(parse_formula("H2O")) / 1000


def example_inputs(
    kind: str = "slow", condition: int = 0, cells: int = 4
) -> tuple[AqueousReactiveTransportSpec, dict[str, dict[str, float]]]:
    """Fixed source declarations for salt, fast acid, and slow precursor cases."""
    if kind not in {"salt", "fast", "slow"} or condition not in (0, 1):
        raise ValueError("unknown authored spatial fixture")
    kc = (1.0, 10.0)[condition]
    ka = kc / (WATER_MASS / WATER_VOLUME)
    kslow = (2.0, 0.5)[condition]
    g_ha = R * TEMPERATURE * log(ka)
    rows = [("H2O", "H2O", 0, "solvent", 0.0, 0.0)]
    if kind == "salt":
        rows += [("M+", "Na", 1, "aqueous", 0.0, 0.0), ("A-", "Cl", -1, "aqueous", 0.0, 0.0)]
    else:
        rows += [
            ("H+", "H", 1, "aqueous", 0.0, 0.0),
            ("A-", "Cl", -1, "aqueous", 0.0, 0.0),
            ("HA", "HCl", 0, "aqueous", -1000.0, g_ha),
        ]
        if kind == "slow":
            rows.append(("S", "HCl", 0, "aqueous", 500.0, g_ha + R * TEMPERATURE * log(kslow)))
    species = []
    for name, formula, charge, phase, h, g in rows:
        composition = parse_formula(formula)
        cp = PropertyCorrelation(
            name + ":cp",
            "cp_polynomial",
            {"a": 10 * sum(composition.values())},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": (280.0, 330.0)},
            "Authored elemental Cp; no measured values",
            {"phase": "liquid"},
        )
        reference = CaloricReference(
            "liquid",
            TEMPERATURE,
            1e5,
            "enthalpy",
            h,
            (h - g) / TEMPERATURE,
            "Authored common reference",
        )
        standard = AqueousStandardState(
            CorrelationSpeciesThermo(name, cp, reference, composition),
            "authored-40D-common",
            "constant-volume",
            (8e4, 2e5),
            SOURCE,
            WATER_VOLUME if phase == "solvent" else 0.0,
        )
        species.append(AqueousSpecies(name, composition, charge, phase, standard, SOURCE))
    reactions = (
        [] if kind == "salt" else [AqueousReaction("acid", {"HA": -1, "H+": 1, "A-": 1}, SOURCE)]
    )
    if kind == "slow":
        reactions.append(AqueousReaction("precursor", {"S": -1, "HA": 1}, SOURCE))
    db = AqueousDatabase(
        "authored-spatial-" + kind,
        (
            AqueousDataSource(
                SOURCE,
                "Authored CC0 functional electrolyte data, not measured parameters",
                "urn:chemworld:authored:40D",
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
            0.1,
            (280.0, 330.0),
            (8e4, 2e5),
        ),
    )
    dp = 1e-9 if kind == "salt" and condition == 0 else 2e-9
    dm = 1e-9
    da = 2 * dp * dm / (dp + dm)
    coefficients = tuple(
        AqueousDiffusionCoefficient(
            s.species_id, dp if s.charge > 0 else dm if s.charge < 0 else da, SOURCE, (280.0, 330.0)
        )
        for s in species
        if s.phase == "aqueous"
    )
    mesh = DiffusionMesh.planar(1e-3, cells, area_m2=1e-3)
    names = tuple(f"cell-{i:02d}" for i in range(cells))
    slow = (
        (AqueousSlowConversion("precursor", "S", "HA", (0.002, 0.008)[condition], SOURCE),)
        if kind == "slow"
        else ()
    )
    spec = AqueousReactiveTransportSpec(
        AqueousEquilibriumSpec(db, () if kind == "salt" else ("acid",)),
        mesh,
        names,
        coefficients,
        slow,
    )
    amounts = {}
    for i, name in enumerate(names):
        volume = float(mesh.cell_volumes_m3[i])
        mode = cos(pi * (i + 0.5) / cells) * sin(pi / (2 * cells)) / (pi / (2 * cells))
        n = {"H2O": volume / WATER_VOLUME}
        if kind == "salt":
            n.update({"M+": volume * 10 * (1 + 0.2 * mode), "A-": volume * 10 * (1 + 0.2 * mode)})
        else:
            total = 20 * (1 + 0.2 * mode)
            ion = 2 * kc * total / (kc + sqrt(kc * kc + 4 * kc * total))
            n.update({"H+": volume * ion, "A-": volume * ion, "HA": volume * (total - ion)})
            if kind == "slow":
                n["S"] = volume * 5 * (1 + 0.15 * mode)
        amounts[name] = n
    return spec, amounts


def run_example() -> dict[str, Any]:
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.aqueous_reactive_transport_world import (
        build_aqueous_reactive_transport_world,
    )

    spec, amounts = example_inputs()
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, amounts))
    try:
        runtime.reset(seed=40)
        step = runtime.execute({"operation": "wait", "duration_s": 100})
        return {
            "transaction_status": step.info["transaction_status"],
            "time_s": runtime._state.ledger.time_s,
            "anchor_amounts_mol": runtime._state.species_amounts,
            "bath_heat_J": runtime._state.ledger.energy_jacket_J,
        }
    finally:
        runtime.close()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), indent=2))
