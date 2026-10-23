"""Native Helmholtz declarations and exact thermodynamic contracts."""

import json
from dataclasses import replace
from math import log, sqrt

import pytest

from chemworld.data.helmholtz import load_helmholtz_fluid
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm


def ideal_fluid():
    return HelmholtzFluid(
        "synthetic-monatomic", "", .04, 8.3, 300, 1000, 100, 1500, 1e9,
        (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0, "a2": 0}),
         HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5})),
        (), "self-authored ideal gas fixture")


def test_ideal_gas_response_and_energy_identities():
    fluid = ideal_fluid()
    state = fluid.state(temperature_K=400, molar_density_mol_m3=20)
    assert state.pressure_Pa == pytest.approx(20*8.3*400)
    assert state.cv_J_mol_K == pytest.approx(1.5*8.3)
    assert state.cp_J_mol_K == pytest.approx(2.5*8.3)
    assert state.speed_of_sound_m_s == pytest.approx(sqrt(5/3*8.3*400/.04))
    assert state.enthalpy_J_mol == pytest.approx(2.5*8.3*400)
    assert state.internal_energy_J_mol == pytest.approx(1.5*8.3*400)
    assert state.entropy_J_mol_K == pytest.approx(8.3*(1.5-log(.02)-1.5*log(.75)))
    assert state.thermal_expansion_K_inv == pytest.approx(1/400)
    assert state.isothermal_compressibility_Pa_inv == pytest.approx(1/state.pressure_Pa)
    assert state.gibbs_energy_J_mol == pytest.approx(
        state.enthalpy_J_mol-400*state.entropy_J_mol_K)
    assert state.helmholtz_energy_J_mol == pytest.approx(
        state.internal_energy_J_mol-400*state.entropy_J_mol_K)
    assert state.to_dict()["stability_scope"] == "homogeneous_local_only"


def test_serialization_keeps_source_and_needs_no_reference_backend():
    fluid = ideal_fluid()
    restored = HelmholtzFluid.from_dict(json.loads(json.dumps(fluid.to_dict())))
    assert restored == fluid
    assert restored.state(temperature_K=400, molar_density_mol_m3=20) == fluid.state(
        temperature_K=400, molar_density_mol_m3=20)


def test_nonunit_prefactor_keeps_raw_source_but_rejects_inconsistent_state():
    fluid = ideal_fluid()
    scaled = replace(fluid, ideal_prefactor=2)
    assert scaled.alpha_derivative("ideal", delta=2, tau=3) == pytest.approx(
        2*fluid.alpha_derivative("ideal", delta=2, tau=3))
    with pytest.raises(ValueError, match="density normalization"):
        scaled.state(temperature_K=400, molar_density_mol_m3=20)


@pytest.mark.parametrize("temperature,density", [(0, 20), (99, 20), (1501, 20), (300, 0)])
def test_physical_input_boundaries(temperature, density):
    with pytest.raises(ValueError):
        ideal_fluid().state(temperature_K=temperature, molar_density_mol_m3=density)


def test_unstable_and_excess_pressure_states_are_not_clipped():
    fluid = replace(ideal_fluid(), residual_terms=(HelmholtzTerm(
        "ResidualHelmholtzPower", {"n": [-2], "d": [1], "t": [0], "l": [0]}),))
    with pytest.raises(ValueError, match="unstable"):
        fluid.state(temperature_K=400, molar_density_mol_m3=400)
    with pytest.raises(ValueError, match="pressure outside"):
        replace(ideal_fluid(), maximum_pressure_Pa=10).state(
            temperature_K=400, molar_density_mol_m3=20)


def test_term_schema_and_derivative_contracts():
    with pytest.raises(ValueError, match="unsupported"):
        HelmholtzTerm.from_dict({"type": "unknown"})
    with pytest.raises(ValueError, match="equal lengths"):
        HelmholtzTerm("ResidualHelmholtzPower", {"n": [1, 2], "d": [1], "t": [0], "l": [0]})
    with pytest.raises(ValueError, match="finite"):
        HelmholtzTerm("IdealGasHelmholtzLead", {"a1": float("nan"), "a2": 0})
    fluid = ideal_fluid()
    assert fluid.alpha_derivative("ideal", delta=2, tau=3, delta_order=3) == pytest.approx(.25)
    assert fluid.alpha_derivative("ideal", delta=2, tau=3, tau_order=3) == pytest.approx(3/27)
    assert fluid.alpha_derivative("ideal", delta=2, tau=3, delta_order=1, tau_order=1) == 0
    with pytest.raises(ValueError, match="total order"):
        fluid.alpha_derivative("ideal", delta=2, tau=3, tau_order=4)


def test_json_selection_units_and_immutable_coefficients():
    eos = {"gas_constant": 8.3, "molar_mass": .04, "T_max": 1500, "p_max": 1e9,
           "pseudo_pure": False,
           "STATES": {"reducing": {"T": 300, "rhomolar": 1000},
                      "sat_min_liquid": {"T": 100}},
           "alpha0": [t.to_dict() for t in ideal_fluid().ideal_terms], "alphar": []}
    data = {"INFO": {"NAME": "test"}, "EOS": [eos, eos]}
    with pytest.raises(ValueError, match="explicit eos_index"):
        load_helmholtz_fluid(data, source="fixture")
    fluid = load_helmholtz_fluid(data, source="fixture", eos_index=1)
    with pytest.raises(TypeError):
        fluid.ideal_terms[0].coefficients["a1"] = 2
    data["EOS"][0] = {**eos, "gas_constant_units": "cal/mol/K"}
    with pytest.raises(ValueError, match="must be"):
        load_helmholtz_fluid(data, source="fixture", eos_index=0)
