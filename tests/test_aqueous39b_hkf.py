"""Revised HKF reference integrity, thermodynamic identities and original consumers."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_hkf import AqueousReferenceBasis
from chemworld.physchem.aqueous_hkf_sources import (
    hkf_aqueous_database,
    hkf_reference_basis,
    hkf_source_records,
    hkf_source_species,
    hkf_source_water,
    hkf_species_from_record,
)
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.water_dielectric import water_dielectric_response
from chemworld.physchem.water_if97 import water_if97_gibbs_state, water_if97_liquid_density
from chemworld.runtime.aqueous_phase_equilibrium import equilibrate_aqueous_phase_inventory
from reference.aqueous39b_cases import inventory, native_inventory


def test_reference_datum_and_roundtrip_preserve_source_quantities():
    db = hkf_aqueous_database()
    payload = json.loads(json.dumps(db.to_dict(), allow_nan=False))
    restored = AqueousDatabase.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.standard_states(573.15, 2e7) == db.standard_states(573.15, 2e7)
    sodium = hkf_source_species("Na+")
    ref = sodium.evaluate_tp(298.15, 1e5)
    assert ref.enthalpy_J_mol == pytest.approx(-57433 * 4.184)
    assert ref.entropy_J_mol_K == pytest.approx(13.96 * 4.184)
    assert sodium.formation_gibbs_ref_J_mol is None


def test_source_inconsistency_is_visible_and_never_silently_repaired():
    original = hkf_source_records()
    with pytest.raises(ValueError, match="inconsistent formation"):
        hkf_source_species("OH-")
    assert original == hkf_source_records()
    original["Na+"]["a"][0] = 0
    assert hkf_source_records()["Na+"]["a"][0] == 0.1839
    assert "OH-" not in {s.species_id for s in hkf_aqueous_database().species}


def test_one_gibbs_temperature_pressure_and_cross_response():
    th = hkf_source_species("Cl-")
    T, P = 573.15, 2e7
    state = th.evaluate_tp(T, P)

    def g(values):
        return th.gibbs(values["T"], values["P"])

    coords = {"T": T, "P": P}
    assert state.enthalpy_J_mol - state.gibbs_J_mol == pytest.approx(T * state.entropy_J_mol_K)
    assert state.molar_volume_m3_mol == pytest.approx(partial_derivative(g, coords, ("P",)))
    assert state.cp_J_mol_K == pytest.approx(-T * partial_derivative(g, coords, ("T", "T")))
    assert partial_derivative(g, coords, ("T", "P")) == pytest.approx(
        partial_derivative(g, coords, ("P", "T")), rel=2e-10
    )
    assert th.omega(T, P) != pytest.approx(th.omega_ref_J_mol, rel=1e-4)


def test_water_datum_conversion_preserves_if97_cp_and_volume():
    water = hkf_source_water()
    ref = water.evaluate_tp(298.15, 1e5)
    assert ref.enthalpy_J_mol == pytest.approx(-285830)
    assert ref.entropy_J_mol_K == pytest.approx(69.95)
    state, upstream = water.evaluate_tp(473.15, 1e7), water_if97_gibbs_state(473.15, 1e7)
    assert state.cp_J_mol_K == pytest.approx(upstream.cp_J_kg_K * water.molar_mass_kg_mol)
    assert state.molar_volume_m3_mol == pytest.approx(
        water.molar_mass_kg_mol / upstream.mass_density_kg_m3
    )
    assert water_if97_liquid_density(473.15, 1e7) == pytest.approx(upstream.mass_density_kg_m3)


def test_dielectric_table_point_and_explicit_domain():
    state = water_dielectric_response(300, 55.31735 * 1000 * 0.018015268)
    assert state.relative_permittivity == pytest.approx(77.74735, abs=2e-5)
    assert state.d_density_kg_m3 > 0 and state.d_temperature_K < 0
    with pytest.raises(ValueError, match="dielectric correlation"):
        water_dielectric_response(220, 1000)
    for T, P in ((650, 3e7), (500, 1e5), (300, 1.1e8)):
        with pytest.raises(ValueError):
            hkf_source_species("Na+").evaluate_tp(T, P)


def test_basis_source_and_coefficient_rejections():
    basis = AqueousReferenceBasis("missing-Na", {"H": 65.34}, -65.34, "test")
    with pytest.raises(ValueError, match="missing elemental"):
        replace(hkf_source_species("Na+"), reference_basis=basis)
    with pytest.raises(ValueError, match="units"):
        hkf_species_from_record(
            "Na+",
            hkf_source_records()["Na+"],
            reference_basis=hkf_reference_basis(),
            source_id="test",
            units="SI",
            consistency_tolerance_J_mol=1,
        )
    db = hkf_aqueous_database()
    with pytest.raises(ValueError, match="undeclared data sources"):
        replace(
            db,
            sources=tuple(
                s for s in db.sources if s.source_id != "nist-janaf-1998-reference-datum"
            ),
        )
    with pytest.raises(ValueError, match="reference basis"):
        replace(db.species[0].standard_state, reference_basis_id="wrong")


def test_basis_identity_cannot_hide_incompatible_entropy_gauge():
    db = hkf_aqueous_database()
    chloride = next(s for s in db.species if s.species_id == "Cl-")
    thermo = chloride.standard_state.thermo
    mismatched = replace(thermo.reference_basis, charge_entropy_J_mol_K=-64)
    changed = replace(
        chloride,
        standard_state=replace(
            chloride.standard_state, thermo=replace(thermo, reference_basis=mismatched)
        ),
    )
    with pytest.raises(ValueError, match="declarations disagree"):
        replace(db, species=tuple(changed if s.species_id == "Cl-" else s for s in db.species))
    with pytest.raises(ValueError, match="explicit elemental and charge"):
        replace(thermo, reference_basis=None)
    with pytest.raises(ValueError, match="charges disagree"):
        replace(chloride, charge=1)


def test_actual_variable_volume_and_original_inventory_candidate():
    db = hkf_aqueous_database()
    cold = aqueous_phase_properties(db, inventory(db), temperature_K=298.15, pressure_Pa=1e5)
    hot = aqueous_phase_properties(db, inventory(db), temperature_K=473.15, pressure_Pa=1e7)
    assert hot.volume_m3 > cold.volume_m3
    spec, ledger, vessel = native_inventory()
    before = ledger.to_dict()
    result = equilibrate_aqueous_phase_inventory(
        ledger, spec, "aqueous", vessel=vessel, temperature_K=473.15, pressure_Pa=1e7
    )
    assert result.success, result.to_dict()
    assert ledger.to_dict() == before
    assert result.ledger.phases["other"].to_dict() == before["other"]
    extent = result.ledger.phases["aqueous"].species_amounts_mol["Cl-test"]
    assert result.heat_duty_J == pytest.approx(2000 * extent, abs=2e-5)
    assert result.boundary_work_J == pytest.approx(0, abs=2e-5)
