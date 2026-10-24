"""Mode ownership and original catalytic consumers of fixed-site hindered G/U."""

import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.examples.native_hindered_adsorbate import (
    example_adsorbate,
    example_mechanism,
    example_world,
)
from chemworld.physchem.catalysis_analysis import (
    CatalyticAnalysisConditions,
    solve_catalytic_steady_state,
    with_catalytic_descriptors,
)
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec
from chemworld.physchem.hindered_adsorbate import HinderedAdsorbateThermo, harmonic_mode_totals
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network_specs import SpeciesSpec
from chemworld.runtime import ChemWorldRuntime


@pytest.mark.parametrize("name", ["soft", "intermediate", "stiff", "ethane-example"])
def test_same_free_energy_generates_entropy_and_caloric_derivative(name):
    model = example_adsorbate(name)
    temperature, step = 500.0, 0.01
    mid = model.evaluate(temperature)
    left, right = (model.evaluate(temperature + s * step) for s in (-1, 1))
    assert -(right.gibbs_J_mol - left.gibbs_J_mol) / (2 * step) == pytest.approx(
        mid.entropy_J_mol_K, rel=1e-7, abs=1e-6
    )
    assert (right.enthalpy_J_mol - left.enthalpy_J_mol) / (2 * step) == pytest.approx(
        mid.cp_J_mol_K, rel=1e-7, abs=1e-6
    )
    restored = HinderedAdsorbateThermo.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.evaluate(temperature) == mid


@pytest.mark.parametrize("barrier", [200 * R * 500, 1e10, 1e16, 1e20])
def test_deep_well_uses_one_stable_bessel_series_and_derivatives(barrier):
    model = replace(
        example_adsorbate(), translation_barrier_J_mol=barrier, rotation_barrier_J_mol=barrier
    )
    t, step = 500.0, 0.1
    mid, left, right = (model.evaluate(at) for at in (t, t - step, t + step))
    # At enormous ZPE, subtraction loses a bounded amount of double precision;
    # the entropy/Cv themselves must still attain the harmonic remainder limit.
    assert -(right.gibbs_J_mol - left.gibbs_J_mol) / (2 * step) == pytest.approx(
        mid.entropy_J_mol_K, rel=1e-5, abs=0.002
    )
    assert (right.enthalpy_J_mol - left.enthalpy_J_mol) / (2 * step) == pytest.approx(
        mid.cp_J_mol_K, rel=1e-5, abs=0.002
    )
    if barrier >= 1e16:
        _, entropy, cv = harmonic_mode_totals(model.vibrational_energies_J_mol, t)
        assert mid.entropy_J_mol_K == pytest.approx(entropy, abs=1e-9)
        assert mid.cp_J_mol_K == pytest.approx(cv, abs=1e-9)


@pytest.mark.parametrize(
    "change",
    [
        {"vibrational_energies_J_mol": (1000, 2000)},
        {"harmonic_mode_ids": ("vib-0", "vib-0", "vib-2")},
        {"hindered_mode_ids": ("vib-0", "y", "z")},
        {"hindered_mode_ids": ("x", "y")},
        {"vibrational_energies_J_mol": (1000, 0, 3000)},
        {"vibrational_energies_J_mol": (1000, 1j, 3000)},
        {"number_of_atoms": 1},
        {"rotational_minima": True},
        {"symmetry_number": 0},
        {"translation_barrier_J_mol": 0},
        {"rotation_barrier_J_mol": -1},
        {"molar_mass_kg_mol": float("nan")},
        {"moment_of_inertia_kg_m2": 0},
        {"site_density_mol_m2": -1},
        {"minimum_temperature_K": 0},
        {"minimum_temperature_K": 1200},
    ],
)
def test_invalid_mode_allocation_or_physical_domain_is_rejected(change):
    with pytest.raises(ValueError):
        replace(example_adsorbate(), **change)


def test_degenerate_frequencies_are_distinct_coordinates_and_symmetry_is_counted_once():
    model = replace(example_adsorbate(), vibrational_energies_J_mol=(1000, 1000, 1000))
    other = replace(model, symmetry_number=2)
    a, b = model.evaluate(600), other.evaluate(600)
    assert b.enthalpy_J_mol == a.enthalpy_J_mol
    assert b.cp_J_mol_K == a.cp_J_mol_K
    assert b.entropy_J_mol_K - a.entropy_J_mol_K == pytest.approx(-R * log(2))
    with pytest.raises(ValueError, match="domain"):
        model.evaluate(1201)


def test_species_composition_and_owned_density_are_part_of_the_model_contract():
    spec = example_mechanism()
    with pytest.raises(ValueError, match="composition"):
        replace(spec.species[1], species=SpeciesSpec("CO_site", "C2H6", phase="surface"))
    for densities in ({}, {"A": 1e-5}, {"A": 2.5e-5, "unknown": 1e-5}):
        with pytest.raises(ValueError, match="density"):
            replace(spec, site_densities_mol_m2=densities)
    encoded = spec.to_dict()
    assert not encoded.get("site_species_complete", False)
    assert encoded["site_densities_mol_m2"] == {"A": 2.5e-5}
    restored = CatalyticMechanismSpec.from_dict(json.loads(json.dumps(encoded)))
    assert restored.to_dict() == encoded
    assert restored.species[1].evaluate(500) == spec.species[1].evaluate(500)


def test_original_descriptor_analysis_rebinds_energy_and_preserves_mode_physics():
    spec = example_mechanism()
    updated = with_catalytic_descriptors(spec, {"binding": 4000})
    old, new = spec.species[1].evaluate(500), updated.species[1].evaluate(500)
    assert new.gibbs_J_mol - old.gibbs_J_mol == pytest.approx(2000)
    assert new.enthalpy_J_mol - old.enthalpy_J_mol == pytest.approx(2000)
    assert new.entropy_J_mol_K == old.entropy_J_mol_K
    assert new.cp_J_mol_K == old.cp_J_mol_K
    assert spec.species[1].adsorbate.descriptors == {"binding": 0}
    encoded = updated.to_dict()
    encoded["species"][1]["adsorbate"]["model"] = "unknown-model"
    with pytest.raises(ValueError, match="unknown"):
        CatalyticMechanismSpec.from_dict(encoded)


def test_original_steady_solver_consumes_hindered_free_energy_in_actual_surface_rates():
    spec = example_mechanism()
    boundary = CatalyticAnalysisConditions(
        temperature_K=600, gas_partial_pressures_Pa={"CO": 5000},
        capacities_mol={"A": 1e-4}, volume_L=1,
    )
    row = solve_catalytic_steady_state(spec, boundary, {"CO_site": 0.2}, target_species="CO")
    delta = spec.species[1].evaluate(600).gibbs_J_mol - spec.species[0].evaluate(600).gibbs_J_mol
    ratio = exp(-delta / (R * 600)) * 5000 / 100000
    assert row["status"] == "converged"
    assert row["stability"] == "stable"
    assert row["coverages"]["CO_site"] == pytest.approx(ratio / (1 + ratio), abs=1e-8)


@pytest.mark.parametrize("isothermal", [True, False])
def test_real_world_owns_finite_stocks_heat_and_exact_json_continuation(isothermal):
    runtime, restored = ChemWorldRuntime(composition=example_world(isothermal=isothermal)), None
    try:
        runtime.reset(seed=53)
        first = runtime.execute({"operation": "wait", "duration_s": 2})
        assert first.info["transaction_status"] == "committed", first.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for world in (runtime, restored):
            reply = world.execute({"operation": "wait", "duration_s": 2})
            assert reply.info["transaction_status"] == "committed", reply.info
        assert runtime._state.to_dict() == restored._state.to_dict()
        state = runtime._state
        bound = state.state_slots.slots[0].values["CO_site"]
        assert 0 < bound < 1e-4
        assert bound != 2e-5
        assert state.species_amounts["CO"] + bound == pytest.approx(0.00202, abs=1e-12)
        assert (state.ledger.energy_jacket_J != 0) is isothermal
        assert (state.temperature_K == 298.15) is isothermal
    finally:
        runtime.close()
        if restored is not None:
            restored.close()
