"""The Helmholtz inventory consumer advances on the original coupled clock."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.coupled_system import (
    CoupledState,
    CoupledSystem,
    HeatFlux,
    ProcessContribution,
)
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics


def thermal_model():
    fluid = HelmholtzFluid("fixture", "", .04, 8.3, 300, 1000, 100, 1500, 1e9,
        (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0, "a2": 0}),
         HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5})),
        (HelmholtzTerm("ResidualHelmholtzPower",
            {"n": [.05], "d": [2], "t": [1], "l": [0]}),), "analytic fixture")
    return HelmholtzIsochoricThermodynamics(fluid, "fluid", 20, 5, 298.15)


@pytest.mark.parametrize("numerical_form", ["ode", "residual"])
def test_actual_shared_heating_conserves_energy_and_support(numerical_form):
    model = thermal_model()
    system = CoupledSystem(model, heat_flux=lambda _: HeatFlux(60), numerical_form=numerical_form)
    start = CoupledState({"fluid": 2}, 500, 20)
    result = system.advance(start, 5)
    final = result.final_state
    assert final is not None
    assert final.temperature_K == pytest.approx(500+300/(2*1.5*8.3+5), abs=1e-7)
    assert model.internal_energy_j(final)-model.internal_energy_j(start) == pytest.approx(300)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-7)
    assert final.time_s == 5 and final.volume_L == 20
    assert start.amounts_mol == {"fluid": 2} and start.temperature_K == 500


def test_nonideal_amount_energy_is_exactly_the_fixed_volume_partial():
    model = thermal_model()
    state = CoupledState({"fluid": 2}, 500, 20)
    expected = 1.5*8.3*500+3*8.3*300*.05*(100/1000)**2
    assert model.partial_inventory_energy_j_mol(state) == pytest.approx(expected)
    assert model.chemical_energy_rate_w(state, {"fluid": -.2}, ()) == pytest.approx(-.2*expected)
    restored = HelmholtzIsochoricThermodynamics.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model and restored.internal_energy_j(state) == model.internal_energy_j(state)


def test_thermal_boundaries_reject_empty_foreign_moving_and_duplicate_energy():
    model = thermal_model()
    state = CoupledState({"fluid": 2}, 500, 20)
    for bad in (replace(state, volume_L=21), replace(state, amounts_mol={"fluid": 0}),
                replace(state, amounts_mol={"other": 2})):
        with pytest.raises(ValueError):
            model.validate_state(bad)
    with pytest.raises(ValueError, match="duplicate"):
        model.chemical_energy_rate_w(state, {"fluid": 0},
            (ProcessContribution(legacy_reaction_energy_W=0),))
    with pytest.raises(ValueError, match="undeclared"):
        model.chemical_energy_rate_w(state, {"other": 1}, ())
