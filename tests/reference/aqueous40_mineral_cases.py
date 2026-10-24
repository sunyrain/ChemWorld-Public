"""Fixed artificial mineral thermochemistry; no real salt accuracy is claimed."""

from math import log

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, VesselRecord
from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousReaction,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_mineral_equilibrium import AqueousMineralEquilibriumSpec
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel, ElectrolyteParameter
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo

R = 8.31446261815324
WATER_MASS = molecular_weight(parse_formula("H2O")) / 1000
SOURCE = "synthetic-mineral40B"


def cases():
    base = {
        "temperature_K": 298.15,
        "pressure_Pa": 1e5,
        "water_kg": 1.0,
        "na": 0.002,
        "k": 0.0,
        "solid_na": 0.0,
        "solid_k": 0.0,
        "complex": 0.0,
        "two": False,
        "association": False,
        "omit_association": False,
        "hydrate": False,
        "family": "ideal",
    }
    return [
        {**base, **update}
        for update in (
            {"id": "absent-undersaturated"},
            {"id": "absent-supersaturated", "na": 0.05},
            {"id": "partial-dissolution", "solid_na": 0.04},
            {"id": "complete-exhaustion", "na": 0.001, "solid_na": 0.002},
            {"id": "common-ion", "two": True, "k": 0.03, "solid_na": 0.03},
            {"id": "two-appear", "two": True, "na": 0.05, "k": 0.06},
            {
                "id": "competing-exhaustion",
                "two": True,
                "k": 0.001,
                "solid_na": 0.04,
                "solid_k": 0.001,
            },
            {"id": "complexation", "na": 0.05, "association": True},
            {
                "id": "omitted-reaction",
                "na": 0.05,
                "association": True,
                "omit_association": True,
                "complex": 0.003,
            },
            {"id": "hydrate", "hydrate": True, "water_kg": 0.25, "solid_na": 0.04},
            {"id": "warm-pressure", "solid_na": 0.04, "temperature_K": 315.0, "pressure_Pa": 2e5},
            {"id": "Davies", "solid_na": 0.04, "family": "Davies"},
        )
    ]


def specification(case):
    hydrate = case["hydrate"]
    mineral = "NaCl.H2O(s)" if hydrate else "NaCl(s)"
    rows = [
        ("H2O", "H2O", 0, "solvent", 0.0, 0.0),
        ("Na+", "Na", 1, "aqueous", 0.0, 0.0),
        ("Cl-", "Cl", -1, "aqueous", 0.0, 0.0),
    ]
    energy = -1500.0 if hydrate else -1000.0
    rows.append(
        (
            mineral,
            "NaClH2O" if hydrate else "NaCl",
            0,
            "mineral",
            energy,
            energy / 298.15 - R * log(1e-4),
        )
    )
    if case["two"]:
        rows.extend(
            (
                ("K+", "K", 1, "aqueous", 0.0, 0.0),
                ("KCl(s)", "KCl", 0, "mineral", -2000.0, -2000.0 / 298.15 - R * log(4e-4)),
            )
        )
    if case["association"]:
        rows.append(("NaCl(aq)", "NaCl", 0, "aqueous", 0.0, R * log(10.0)))
    species = []
    for key, formula, charge, phase, enthalpy, entropy in rows:
        composition = parse_formula(formula)
        atoms = sum(composition.values())
        caloric_phase = "solid" if phase == "mineral" else "liquid"
        cp = PropertyCorrelation(
            key + "-cp",
            "cp_polynomial",
            {"a": 10.0 * atoms},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": (275.0, 350.0)},
            "Artificial elemental-additive Cp",
            {"phase": caloric_phase},
        )
        reference = CaloricReference(
            caloric_phase,
            298.15,
            1e5,
            "enthalpy",
            enthalpy,
            entropy,
            "Artificial shared elemental gauge",
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
    reaction = {mineral: -1, "Na+": 1, "Cl-": 1, **({"H2O": 1} if hydrate else {})}
    reactions = [AqueousReaction("dissolve-Na", reaction, SOURCE)]
    if case["two"]:
        reactions.append(AqueousReaction("dissolve-K", {"KCl(s)": -1, "K+": 1, "Cl-": 1}, SOURCE))
    if case["association"]:
        reactions.append(
            AqueousReaction("associate", {"Na+": -1, "Cl-": -1, "NaCl(aq)": 1}, SOURCE)
        )
    parameters = (
        {}
        if case["family"] == "ideal"
        else {
            key: ElectrolyteParameter(value, "(kg/mol)^0.5", SOURCE, (275.0, 350.0), (1e5, 1e6))
            for key, value in (("A", 0.509), ("b", 0.3))
        }
    )
    activities = ElectrolyteActivityModel(
        SOURCE + case["family"],
        case["family"],
        tuple((s.species_id, s.charge) for s in species if s.phase == "aqueous"),
        SOURCE,
        0.2,
        (275.0, 350.0),
        (1e5, 1e6),
        parameters,
    )
    source = AqueousDataSource(
        SOURCE,
        "Artificial 40B fixtures, not mineral parameters",
        "urn:chemworld:synthetic:40B",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
    )
    db = AqueousDatabase(
        SOURCE + case["id"], (source,), tuple(species), tuple(reactions), activities
    )
    selected = tuple(
        r.reaction_id
        for r in reactions
        if not (case["omit_association"] and r.reaction_id == "associate")
    )
    return AqueousMineralEquilibriumSpec(db, selected)


def inventory(case, spec):
    result = dict.fromkeys((s.species_id for s in spec.database.species), 0.0)
    result.update(
        {"H2O": case["water_kg"] / WATER_MASS, "Na+": case["na"], "Cl-": case["na"] + case["k"]}
    )
    result["NaCl.H2O(s)" if case["hydrate"] else "NaCl(s)"] = case["solid_na"]
    if case["two"]:
        result.update({"K+": case["k"], "KCl(s)": case["solid_k"]})
    if case["association"]:
        result["NaCl(aq)"] = case["complex"]
    return result


def owned_inventory(case, spec=None):
    spec = spec or specification(case)
    amounts = inventory(case, spec)
    liquid = {
        s.species_id: amounts[s.species_id] for s in spec.database.species if s.phase != "mineral"
    }
    liquid_properties = aqueous_phase_properties(
        spec.database, liquid, temperature_K=case["temperature_K"], pressure_Pa=case["pressure_Pa"]
    )
    phases = {
        "water": PhaseRecord(
            "water",
            "cell",
            "aqueous",
            liquid_properties.volume_m3 * 1000,
            liquid,
            selected=True,
            metadata={"carrier_representation": "species_inventory", "user_tag": "keep"},
        )
    }
    phase_ids = {k: "original:" + k for k in spec.mineral_ids}
    for species in spec.database.species:
        if species.phase == "mineral":
            key = phase_ids[species.species_id]
            n = amounts[species.species_id]
            phases[key] = PhaseRecord(
                key,
                "cell",
                "solid",
                n * species.standard_state.molar_volume_m3_mol * 1000,
                {species.species_id: n},
            )
    phases["other"] = PhaseRecord("other", "other-cell", "solid", 0.1, {"unrelated": 1.0})
    vessel = VesselRecord(
        "cell",
        "TP-cell",
        2.0,
        350.0,
        1e6,
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )
    return spec, PhaseLedger(phases), vessel, phase_ids
