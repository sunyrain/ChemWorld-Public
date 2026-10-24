"""Authored common-ion coupling; formulas identify elements, not measured materials."""

from dataclasses import replace
from math import log
from typing import Any, Literal

from chemworld.examples.aqueous_headspace import TREF, VOLUME
from chemworld.examples.aqueous_headspace import example_inputs as gas_inputs
from chemworld.physchem.aqueous_database import AqueousReaction, AqueousSpecies
from chemworld.physchem.aqueous_headspace_minerals import (
    AqueousHeadspaceMineralSpec,
    prepare_aqueous_headspace_mineral_state,
)
from chemworld.physchem.aqueous_mineral_kinetics import MineralRateLaw
from chemworld.physchem.aqueous_reaction_equilibrium import AqueousEquilibriumSpec
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world


def example_inputs(
    kind: str = "uptake", condition: int = 0
) -> tuple[AqueousHeadspaceMineralSpec, dict[str, float], dict[str, MineralGeometry]]:
    if kind not in {
        "uptake",
        "degassing",
        "exhaustion",
        "unseeded",
        "double",
        "zero_gas",
        "zero_mineral",
    }:
        raise ValueError("unknown authored headspace mineral fixture")
    gas, amounts = gas_inputs("double" if kind == "double" else "acid", condition)
    db = gas.equilibrium.database
    source = db.sources[0].source_id
    aqueous = next(s for s in db.species if s.species_id == "H+")
    species, reactions = list(db.species), list(db.reactions)
    mineral_ids = ["MA(s)", "KA(s)"] if kind == "double" else ["MA(s)"]
    geometries, laws = {}, []
    for m in mineral_ids:
        cation, formula, ksp, volume = (
            ("M+", "NaCl", 5e-4, 1e-3) if m == "MA(s)" else ("K+", "KCl", 8e-4, 5e-4)
        )
        for key, composition, charge, phase, h, g, v in (
            (cation, parse_formula("Na" if m == "MA(s)" else "K"), 1, "aqueous", 0.0, 0.0, 0.0),
            (m, parse_formula(formula), 0, "mineral", -2000.0, R * TREF * log(ksp), volume),
        ):
            thermo_phase: Literal["liquid", "solid"] = "solid" if phase == "mineral" else "liquid"
            thermo = aqueous.standard_state.thermo
            thermo = replace(
                thermo,
                species_id=key,
                composition=composition,
                heat_capacity=replace(
                    thermo.heat_capacity,
                    correlation_id=key + ":cp",
                    coefficients={"a": 10 * sum(composition.values())},
                    metadata={"phase": thermo_phase},
                ),
                reference=replace(
                    thermo.reference,
                    phase=thermo_phase,
                    energy_J_mol=h,
                    entropy_J_mol_K=(h - g) / TREF,
                ),
            )
            standard = replace(aqueous.standard_state, thermo=thermo, molar_volume_m3_mol=v)
            species.append(AqueousSpecies(key, composition, charge, phase, standard, source))
            amounts[key] = 0.0
        reactions.append(AqueousReaction("dissolve:" + m, {m: -1, cation: 1, "Cl-": 1}, source))
        amount = 1e-5 if kind == "exhaustion" else 0.0 if kind == "unseeded" else 0.004
        amounts[m] = amount
        dissolved = 0.0 if kind == "exhaustion" else 0.002 if m == "MA(s)" else 0.001
        amounts[cation] = dissolved
        amounts["Cl-"] += dissolved
        geometries[m] = MineralGeometry(0.01, amount if amount else 0.004)
        laws.append(MineralRateLaw(m, 0.0 if kind == "zero_mineral" else 0.002, 1200.0))
    if kind == "degassing":
        amounts["A(g)"] = 0.0
        amounts["A(aq)"] += 0.004
    db = replace(
        db,
        database_id="authored-headspace-minerals-" + kind,
        species=tuple(species),
        reactions=tuple(reactions),
        activity_model=replace(
            db.activity_model,
            charges=tuple((s.species_id, s.charge) for s in species if s.phase == "aqueous"),
        ),
    )
    transfers = (
        tuple(replace(t, coefficient_mol_s=0.0) for t in gas.transfers)
        if kind == "zero_gas"
        else gas.transfers
    )
    spec = AqueousHeadspaceMineralSpec(
        AqueousEquilibriumSpec(db, ("acid",)),
        transfers,
        tuple("dissolve:" + m for m in mineral_ids),
        tuple(laws),
        VOLUME,
        2 * VOLUME,
        gas.temperature_K,
    )
    return spec, prepare_aqueous_headspace_mineral_state(spec, amounts), geometries


def run_example() -> dict[str, Any]:
    from chemworld.runtime import ChemWorldRuntime

    spec, amounts, geometries = example_inputs()
    runtime = ChemWorldRuntime(
        composition=build_aqueous_headspace_world(spec, amounts, mineral_geometries=geometries)
    )
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
