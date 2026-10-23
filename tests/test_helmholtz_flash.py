"""Pure-fluid manifold closure with a self-authored polynomial energy fixture."""

from dataclasses import replace
from itertools import combinations

import pytest

from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_flash import (
    HelmholtzTwoPhaseState,
    coexistence_derivatives,
    flash_at_pressure,
    saturation_at_pressure,
    solve_two_phase_state,
)
from chemworld.physchem.helmholtz_solvers import HelmholtzAncillary, HelmholtzPhaseModel
from chemworld.physchem.helmholtz_terms import HelmholtzTerm


def polynomial_fluid():
    fluid = HelmholtzFluid("polynomial", "", .04, 8.3, 300, 1000, 100, 1500, 1e9,
        (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0, "a2": 0}),
         HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5})),
        (HelmholtzTerm("ResidualHelmholtzPower",
                      {"n": [1/60, -2/3], "d": [4, 1], "t": [0, 1], "l": [0, 0]}),),
        "self-authored polynomial repulsion and mean-field attraction")
    return HelmholtzPhaseModel(fluid, 300, 1000, .4*1000*8.3*300, {
        "rhoL": HelmholtzAncillary("rhoLnoexp", 300, 1000, 100, 300, (2,), (.5,)),
        "rhoV": HelmholtzAncillary("rhoV", 300, 1000, 100, 300, (-4,), (.5,)),
        "pS": HelmholtzAncillary("pV", 300, .4*1000*8.3*300, 100, 300, (-4,), (1,), True),
    })


def mixture():
    model = polynomial_fluid()
    report = model.saturation(240)
    assert report.success, report.failure_reason
    return model, HelmholtzTwoPhaseState(report.liquid, report.gas, .35)


def values(state):
    return {"T": state.temperature_K, "P": state.pressure_Pa,
            "rho": state.molar_density_mol_m3, "H": state.enthalpy_J_mol,
            "U": state.internal_energy_J_mol, "S": state.entropy_J_mol_K,
            "Q": state.vapor_fraction}


def test_volume_and_caloric_mixing_and_endpoint_validation():
    _, state = mixture()
    assert state.enthalpy_J_mol == pytest.approx(
        state.internal_energy_J_mol+state.pressure_Pa/state.molar_density_mol_m3)
    assert state.molar_density_mol_m3 != pytest.approx(
        .65*state.liquid.molar_density_mol_m3+.35*state.gas.molar_density_mol_m3)
    with pytest.raises(ValueError, match="zero and one"):
        replace(state, vapor_fraction=-1e-12)
    with pytest.raises(ValueError, match="coexistence"):
        replace(state, gas=replace(state.gas, temperature_K=241))
    with pytest.raises(ValueError, match="identities"):
        replace(state, gas=replace(state.gas, fluid_id="other"))
    assert "cp_J_mol_K" not in state.to_dict()


@pytest.mark.parametrize("pair", [p for p in combinations(("T", "P", "rho", "H", "U", "S", "Q"), 2)
                                  if set(p) != {"T", "P"}])
def test_all_twenty_independent_coexistence_pairs(pair):
    model, target = mixture()
    report = solve_two_phase_state(model, {k: values(target)[k] for k in pair},
                                   initial_temperature_K=239, initial_vapor_fraction=.36)
    assert report.success, (pair, report.failure_reason)
    assert report.state.temperature_K == pytest.approx(240, abs=1e-7)
    assert report.state.vapor_fraction == pytest.approx(.35, abs=1e-8)
    assert not report.diagnostics["global_uniqueness_claimed"]


def test_coexistence_response_matches_directional_perturbations():
    model, state = mixture()
    derivatives = coexistence_derivatives(model, state)
    plus = solve_two_phase_state(model, {"T": 240.001, "Q": .35}).state
    minus = solve_two_phase_state(model, {"T": 239.999, "Q": .35}).state
    for key in values(state):
        assert derivatives[key][0] == pytest.approx(
            (values(plus)[key]-values(minus)[key])/.002, rel=2e-7, abs=1e-8)
    with pytest.raises(ValueError, match="different fluid"):
        coexistence_derivatives(replace(model, fluid=replace(model.fluid, fluid_id="other")), state)


@pytest.mark.parametrize("key", ["H", "U", "S", "rho"])
def test_pressure_flash_selects_mixture_and_both_homogeneous_branches(key):
    model, state = mixture()
    report = flash_at_pressure(model, pressure_Pa=state.pressure_Pa,
                               coordinate=key, value=values(state)[key])
    assert report.success and report.phase == "two_phase"
    assert report.state.vapor_fraction == pytest.approx(.35)
    for T, phase in ((230, "liquid"), (250, "gas")):
        endpoint = model.solve_tp(temperature_K=T, pressure_Pa=state.pressure_Pa).state
        target = {"H": endpoint.enthalpy_J_mol, "U": endpoint.internal_energy_J_mol,
                  "S": endpoint.entropy_J_mol_K, "rho": endpoint.molar_density_mol_m3}[key]
        report = flash_at_pressure(model, pressure_Pa=state.pressure_Pa,
                                   coordinate=key, value=target)
        assert report.success, report.failure_reason
        assert report.phase == phase
        assert report.state.temperature_K == pytest.approx(T)


def test_unsupported_domains_are_not_fabricated_equilibrium():
    model, state = mixture()
    assert not solve_two_phase_state(model, {"T": 240, "P": state.pressure_Pa}).success
    assert not solve_two_phase_state(model, {"T": 240, "rho": 1}).success
    with pytest.raises(ValueError, match="explicit"):
        solve_two_phase_state(model, {"U": 100, "rho": 50})
    with pytest.raises(ValueError, match="pseudo-pure"):
        solve_two_phase_state(replace(model, fluid=replace(model.fluid, pseudo_pure=True)),
                              {"T": 240, "Q": .5})
    with pytest.raises(ValueError, match="critical pressure"):
        saturation_at_pressure(model, model.critical_pressure_Pa)
    with pytest.raises(ValueError, match="zero and one"):
        solve_two_phase_state(model, {"T": 240, "Q": 1.01})


def test_source_nominal_critical_density_does_not_truncate_coexistence_roots():
    model, original = mixture()
    changed = replace(model, critical_molar_density_mol_m3=2000)
    report = changed.saturation(240)
    assert report.success
    assert report.liquid.molar_density_mol_m3 == pytest.approx(original.liquid.molar_density_mol_m3)
    assert report.gas.molar_density_mol_m3 == pytest.approx(original.gas.molar_density_mol_m3)


def test_zero_source_fit_limit_and_explicit_initial_guess_extrapolation():
    model, _ = mixture()
    ancillary = replace(model.ancillaries["pS"], minimum_temperature_K=0)
    assert ancillary.value(240) > 0
    with pytest.raises(ValueError, match="positive"):
        ancillary.value(0)
    changed = replace(model, ancillaries={k: replace(v, minimum_temperature_K=250)
                                          for k, v in model.ancillaries.items()})
    report = changed.saturation(240)
    assert report.success
    assert set(report.diagnostics["extrapolated_ancillary_initial_guesses"]) == {
        "rhoL", "rhoV", "pS"}
    with pytest.raises(ValueError, match="domain"):
        changed.ancillaries["pS"].value(240)
