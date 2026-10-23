"""Equilibrium latent energy, with one physical inventory and one clock."""

import json
from dataclasses import replace

import pytest
from tests.test_helmholtz_flash import mixture

from chemworld.physchem.coupled_system import CoupledState, CoupledSystem, HeatFlux
from chemworld.physchem.helmholtz_equilibrium_thermal import (
    HelmholtzEquilibriumIsochoricThermodynamics,
)
from chemworld.physchem.helmholtz_flash import HelmholtzTwoPhaseState
from chemworld.physchem.solver_backend import ODESolverPolicy


def initial_model():
    model, state = mixture()
    volume = 2000/state.molar_density_mol_m3
    thermo = HelmholtzEquilibriumIsochoricThermodynamics(model, "fluid", volume)
    return thermo, CoupledState({"fluid": 2}, 240, volume)


def test_fixed_density_latent_heat_capacity_and_inventory_partial():
    thermo, initial = initial_model()
    dT, dn = 1e-3, 1e-5
    cv = (thermo.internal_energy_j(replace(initial, temperature_K=240+dT))
          -thermo.internal_energy_j(replace(initial, temperature_K=240-dT)))/(2*dT)
    un = (thermo.internal_energy_j(replace(initial, amounts_mol={"fluid": 2+dn}))
          -thermo.internal_energy_j(replace(initial, amounts_mol={"fluid": 2-dn})))/(2*dn)
    assert thermo.heat_capacity_j_k(initial) == pytest.approx(cv, rel=1e-8)
    assert thermo.partial_inventory_energy_j_mol(initial) == pytest.approx(un, rel=1e-8)
    mixture = thermo.molar_state(initial)
    wrong_cv = 2*(.65*mixture.liquid.cv_J_mol_K+.35*mixture.gas.cv_J_mol_K)
    assert thermo.heat_capacity_j_k(initial) != pytest.approx(wrong_cv)
    restored = HelmholtzEquilibriumIsochoricThermodynamics.from_dict(
        json.loads(json.dumps(thermo.to_dict())))
    assert restored == thermo


@pytest.mark.parametrize("form", ["ode", "residual"])
def test_actual_equilibrium_heating_preserves_finite_energy(form):
    thermo, initial = initial_model()
    system = CoupledSystem(thermo, heat_flux=lambda _: HeatFlux(60), numerical_form=form,
        policy=ODESolverPolicy("equilibrium-heat", method="BDF", rtol=1e-11, atol=1e-13))
    result = system.advance(initial, 5)
    final = result.final_state
    assert isinstance(thermo.molar_state(final), HelmholtzTwoPhaseState)
    assert final.temperature_K > 240
    assert thermo.molar_state(final).vapor_fraction > .35
    assert thermo.internal_energy_j(final)-thermo.internal_energy_j(initial) == pytest.approx(
        300, abs=1e-6)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-6)


def test_equilibrium_model_uses_same_homogeneous_energy_outside_coexistence():
    thermo, initial = initial_model()
    hot = replace(initial, temperature_K=500)
    expected = thermo.phase_model.fluid.state(temperature_K=500,
        molar_density_mol_m3=2/(initial.volume_L*1e-3))
    assert thermo.molar_state(hot) == expected
    assert thermo.heat_capacity_j_k(hot) == pytest.approx(2*expected.cv_J_mol_K)
    with pytest.raises(ValueError, match="changing volume"):
        thermo.molar_state(replace(initial, volume_L=10))
    with pytest.raises(ValueError, match="empty phase"):
        thermo.molar_state(replace(initial, amounts_mol={"fluid": 0}))
