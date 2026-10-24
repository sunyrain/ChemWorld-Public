"""Synthetic single-liquid closures, with explicitly artificial thermochemistry."""

from dataclasses import replace
from math import log

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousReaction,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_reaction_equilibrium import AqueousEquilibriumSpec
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_reports import R_J_PER_MOL_K, CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo
from reference.aqueous39_fixtures import P_BOUNDS, SOURCE, T_BOUNDS, activity

WATER_MASS = molecular_weight(parse_formula("H2O")) / 1000
KW = 0.04
KA = 1e-4


def make_database(family="ideal", *, network="acid", redundant=False):
    # Standard entropy values enforce temperature-independent artificial K.
    rows = [
        ("H2O", "H2O", 0, "solvent", 0.0),
        ("H+", "H", 1, "aqueous", 0.0),
    ]
    if network in ("water", "mixed"):
        rows.append(("OH-", "OH", -1, "aqueous", R_J_PER_MOL_K * log(KW)))
    if network in ("acid", "mixed"):
        rows.extend(
            [
                ("HA", "HCl", 0, "aqueous", -R_J_PER_MOL_K * log(KA)),
                ("A-", "Cl", -1, "aqueous", 0.0),
            ]
        )
    species = []
    for key, formula, charge, phase, entropy in rows:
        composition = parse_formula(formula)
        atoms = sum(composition.values())
        cp = PropertyCorrelation(
            key + "-cp",
            "cp_polynomial",
            {"a": 10.0 * atoms},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": T_BOUNDS},
            "Synthetic elemental-additive Cp gives zero reaction DeltaCp",
            {"phase": "liquid"},
        )
        reference = CaloricReference(
            "liquid", 298.15, 1e5, "enthalpy", 0.0, entropy, "Synthetic common elemental gauge"
        )
        thermo = CorrelationSpeciesThermo(key, cp, reference, composition)
        # Elemental-additive V gives zero reaction DeltaV and finite total volume.
        state = AqueousStandardState(
            thermo, "synthetic-40-gauge", "constant-volume", P_BOUNDS, SOURCE, 1e-6 * atoms
        )
        species.append(AqueousSpecies(key, parse_formula(formula), charge, phase, state, SOURCE))
    reactions = []
    if network in ("water", "mixed"):
        reactions.append(AqueousReaction("water", {"H2O": -1, "H+": 1, "OH-": 1}, SOURCE))
    if network in ("acid", "mixed"):
        reactions.append(AqueousReaction("acid", {"HA": -1, "H+": 1, "A-": 1}, SOURCE))
    if redundant:
        first = reactions[0]
        reactions.append(
            AqueousReaction("reverse", {k: -2 * v for k, v in first.stoichiometry.items()}, SOURCE)
        )
    source = AqueousDataSource(
        SOURCE,
        "Synthetic equilibrium fixture, not material parameters",
        "urn:chemworld:synthetic:aqueous40",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
    )
    charges = tuple((s.species_id, s.charge) for s in species if s.phase == "aqueous")
    return AqueousDatabase(
        "synthetic-aqueous40-" + network + "-" + family,
        (source,),
        tuple(species),
        tuple(reactions),
        activity(family, charges),
    )


def model(family="ideal", **kwargs):
    db = make_database(family, **kwargs)
    return AqueousEquilibriumSpec(db, tuple(r.reaction_id for r in db.reactions))


def feed(spec, water_kg=1.0, acid_molality=0.1):
    result = {s.species_id: 0.0 for s in spec.database.species}
    result["H2O"] = water_kg / WATER_MASS
    if "HA" in result:
        result["HA"] = acid_molality * water_kg
    return result


def add_inert_species(spec):
    db = spec.database
    template = db.species[0]
    composition = parse_formula("C")
    thermo = replace(template.standard_state.thermo, species_id="inert-C", composition=composition)
    state = replace(template.standard_state, thermo=thermo)
    species = AqueousSpecies("inert-C", composition, 0, "aqueous", state, SOURCE)
    charges = (*db.activity_model.charges, ("inert-C", 0))
    return replace(
        spec,
        database=replace(
            db,
            species=(*db.species, species),
            activity_model=activity(db.activity_model.family, charges),
        ),
    )


def isomer_model(*, blocked=False, unstable=False):
    db = make_database("ideal")
    water, template = db.species[0], next(s for s in db.species if s.species_id == "HA")
    isomers = []
    for key in ("A", "B", "C") if blocked else ("A", "B"):
        thermo = replace(template.standard_state.thermo, species_id=key)
        isomers.append(
            replace(
                template,
                species_id=key,
                standard_state=replace(template.standard_state, thermo=thermo),
            )
        )
    stoich = {"A": -1, "B": -1, "C": 2} if blocked else {"A": -1, "B": 1}
    reaction = AqueousReaction("isomer", stoich, SOURCE)
    activities = activity("SIT" if unstable else "ideal", tuple((s.species_id, 0) for s in isomers))
    if unstable:
        parameters = {**activities.parameters, "A": replace(activities.parameters["A"], value=0)}
        interactions = tuple(
            replace(
                item,
                parameters={
                    "epsilon": replace(
                        item.parameters["epsilon"], value=2.0 if item.species == ("A", "B") else 0.0
                    )
                },
            )
            for item in activities.interactions
        )
        activities = replace(activities, parameters=parameters, interactions=interactions)
    return AqueousEquilibriumSpec(
        replace(db, species=(water, *isomers), reactions=(reaction,), activity_model=activities),
        ("isomer",),
    )


def complexation_model():
    db = make_database("ideal")
    water = db.species[0]
    template = next(s for s in db.species if s.species_id == "HA")
    species = []
    for key, formula, charge, entropy in (
        ("M+", "Na", 1, 0.0),
        ("L-", "Cl", -1, 0.0),
        ("ML", "NaCl", 0, R_J_PER_MOL_K * log(10)),
    ):
        composition = parse_formula(formula)
        old = template.standard_state.thermo
        thermo = replace(
            old,
            species_id=key,
            composition=composition,
            heat_capacity=replace(
                old.heat_capacity, coefficients={"a": 10.0 * sum(composition.values())}
            ),
            reference=replace(old.reference, entropy_J_mol_K=entropy),
        )
        state = replace(
            template.standard_state,
            thermo=thermo,
            molar_volume_m3_mol=1e-6 * sum(composition.values()),
        )
        species.append(AqueousSpecies(key, composition, charge, "aqueous", state, SOURCE))
    reaction = AqueousReaction("complex", {"M+": -1, "L-": -1, "ML": 1}, SOURCE)
    activities = activity("ideal", tuple((s.species_id, s.charge) for s in species))
    return AqueousEquilibriumSpec(
        replace(db, species=(water, *species), reactions=(reaction,), activity_model=activities),
        ("complex",),
    )
