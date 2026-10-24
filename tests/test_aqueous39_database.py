"""Actual source-aware species, reaction and common property consumers."""

import json
from dataclasses import replace
from math import log

import pytest

from chemworld.physchem.aqueous_database import AqueousDatabase, AqueousReaction
from chemworld.physchem.property_reports import R_J_PER_MOL_K
from chemworld.physchem.thermochemistry import NASA7SpeciesThermo, NASA7TemperatureSegment
from reference.aqueous39_fixtures import database, inventory


def test_database_json_roundtrip_and_standard_state_temperature_pressure():
    db = database()
    payload = json.loads(json.dumps(db.to_dict(), allow_nan=False))
    restored = AqueousDatabase.from_dict(payload)
    assert json.loads(json.dumps(restored.to_dict())) == payload
    for species in db.species:
        ref = species.standard_state.thermo.evaluate(298.15)
        cp = ref.cp_J_mol_K
        for t in (280.0, 298.15, 330.0):
            for p in (1e5, 1e6):
                actual = species.standard_state.evaluate(t, p)
                shift = (species.standard_state.molar_volume_m3_mol or 0) * (p - 1e5)
                assert actual.enthalpy_J_mol == pytest.approx(
                    ref.enthalpy_J_mol + cp * (t - 298.15) + shift
                )
                assert actual.entropy_J_mol_K == pytest.approx(
                    ref.entropy_J_mol_K + cp * log(t / 298.15)
                )
                assert actual.gibbs_J_mol == pytest.approx(
                    actual.enthalpy_J_mol - t * actual.entropy_J_mol_K
                )


def test_actual_chemical_potentials_reactions_solvent_and_property_consumer():
    db = database()
    amounts = inventory(db)
    report = db.chemical_potentials(
        amounts, temperature_K=330, pressure_Pa=1e6, gas_fugacities_Pa={"CO2g": 2e4}
    )
    assert report.solvent_mass_kg == pytest.approx(1)
    assert report.log_activities["CO2g"] == pytest.approx(log(0.2))
    assert report.log_activities["Calcite"] == 0
    assert report.log_activities["Aragonite"] == 0
    for result in db.reaction_affinities(report).values():
        assert not result["absent_species"]
        assert result["reaction_gibbs_J_mol"] == pytest.approx(
            result["delta_gibbs_J_mol"] + R_J_PER_MOL_K * 330 * result["log_reaction_quotient"]
        )
        assert result["affinity_J_mol"] == -result["reaction_gibbs_J_mol"]
        assert result["log_equilibrium_constant"] == pytest.approx(
            -result["delta_gibbs_J_mol"] / (R_J_PER_MOL_K * 330)
        )
    doubled = db.chemical_potentials(
        {k: 2 * v for k, v in amounts.items()},
        temperature_K=330,
        pressure_Pa=1e6,
        gas_fugacities_Pa={"CO2g": 2e4},
    )
    assert doubled.chemical_potentials_J_mol == report.chemical_potentials_J_mol
    liquid = {
        s.species_id: amounts[s.species_id] for s in db.species if s.phase in ("solvent", "aqueous")
    }
    common = db.property_report(liquid, temperature_K=330, pressure_Pa=1e6)
    assert common.valid
    assert (
        common.component_properties["chemical_potential"]["H+"].value
        == report.chemical_potentials_J_mol["H+"]
    )
    assert common.model_details["solvent_activity"] < 1
    assert json.loads(json.dumps(common.to_dict(), allow_nan=False))


def test_missing_fugacity_actual_solvent_and_absent_species():
    db = database()
    amounts = inventory(db)
    with pytest.raises(ValueError, match="positive fugacity"):
        db.chemical_potentials(amounts, temperature_K=298.15, pressure_Pa=1e5)
    with pytest.raises(ValueError, match="solvent inventory"):
        db.chemical_potentials({**amounts, "H2O": 0}, temperature_K=298.15, pressure_Pa=1e5)
    report = db.chemical_potentials({"H2O": amounts["H2O"]}, temperature_K=298.15, pressure_Pa=1e5)
    assert report.chemical_potentials_J_mol["Na+"] is None
    assert db.reaction_affinities(report)["water"]["affinity_J_mol"] is None
    assert json.loads(json.dumps(report.to_dict(), allow_nan=False))


def test_database_missing_sources_units_basis_and_exact_conservation():
    db = database()
    with pytest.raises(ValueError, match="unbalanced"):
        replace(
            db,
            reactions=(
                AqueousReaction("bad", {"H2O": -1, "H+": 2, "OH-": 1}, db.sources[0].source_id),
            ),
        )
    with pytest.raises(ValueError, match="undeclared data sources"):
        replace(db, sources=(replace(db.sources[0], source_id="different"),))
    first = db.species[0]
    with pytest.raises(ValueError, match="chemical-potential basis"):
        replace(
            db,
            species=(
                replace(
                    first,
                    standard_state=replace(
                        first.standard_state, reference_basis_id="different-gauge"
                    ),
                ),
                *db.species[1:],
            ),
        )
    payload = db.to_dict()
    payload["units"]["pressure"] = "bar"
    with pytest.raises(ValueError, match="noncanonical units"):
        AqueousDatabase.from_dict(payload)
    with pytest.raises(ValueError, match="pressure dependence"):
        replace(
            first.standard_state, pressure_model="reference-only", molar_volume_m3_mol=None
        ).evaluate(298.15, 2e5)


def test_nasa7_shared_species_thermo_codec():
    db = database()
    gas = db.species[-1]
    nasa = NASA7SpeciesThermo(
        "CO2g",
        (NASA7TemperatureSegment(275, 350, (3.5, 0, 0, 0, 0, -1000, 2)),),
        1e5,
        dict(gas.composition),
        "synthetic NASA7",
    )
    db = replace(
        db,
        species=(
            *db.species[:-1],
            replace(gas, standard_state=replace(gas.standard_state, thermo=nasa)),
        ),
    )
    restored = AqueousDatabase.from_dict(json.loads(json.dumps(db.to_dict())))
    assert restored.standard_states(300, 1e5)["CO2g"] == db.standard_states(300, 1e5)["CO2g"]
