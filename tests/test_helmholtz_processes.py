"""Pressure stages consume original material ports and finite receiving energy."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    MaterialPort,
    ProcessContribution,
)
from chemworld.physchem.helmholtz_processes import HelmholtzPressureStage, HelmholtzStream
from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics
from chemworld.physchem.solver_backend import ODESolverPolicy
from test_helmholtz_flash import polynomial_fluid


def stage(mode):
    model = polynomial_fluid()
    temperature, pressure = (240, 800000) if mode == "throttle" else (400, 200000)
    upstream = model.solve_tp(temperature_K=temperature, pressure_Pa=pressure).state
    target = 400000 if mode == "compressor" else 100000
    return HelmholtzPressureStage(model, mode, target, 1 if mode == "throttle" else .8), upstream


@pytest.mark.parametrize("mode", ["throttle", "compressor", "expander"])
def test_pressure_stages_preserve_mass_energy_and_entropy_direction(mode):
    device, upstream = stage(mode)
    result = device.evaluate(HelmholtzStream(upstream, .01))
    assert result.success, result.failure_reason
    assert result.outlet.molar_rate_mol_s == .01
    assert result.outlet.state.pressure_Pa == pytest.approx(device.outlet_pressure_Pa)
    assert result.diagnostics["energy_residual_W"] == pytest.approx(0, abs=1e-7)
    assert result.diagnostics["entropy_generation_W_K"] >= 0
    if mode == "throttle":
        assert result.shaft_work_W == 0
        liquid, gas = result.outlet.separate()
        assert liquid.molar_rate_mol_s+gas.molar_rate_mol_s == pytest.approx(.01)
        assert liquid.enthalpy_rate_W+gas.enthalpy_rate_W == pytest.approx(
            result.outlet.enthalpy_rate_W)
    else:
        assert result.shaft_work_W*(1 if mode == "compressor" else -1) > 0
    restored = HelmholtzPressureStage.from_dict(json.loads(json.dumps(device.to_dict())))
    assert restored == device


@pytest.mark.parametrize("numerical_form", ["ode", "residual"])
def test_flash_liquid_and_vapor_feed_real_finite_inventory_once(numerical_form):
    device, upstream = stage("throttle")
    stage_result = device.evaluate(HelmholtzStream(upstream, .01))
    liquid, gas = stage_result.outlet.separate()
    receiver = HelmholtzIsochoricThermodynamics(device.phase_model.fluid, "fluid", 2000)
    ports = (MaterialPort("liquid", "fluid", "separator", "vessel", "in"),
             MaterialPort("gas", "fluid", "separator", "vessel", "in"))
    contributions = [s.contribution(p.port_id) for s, p in zip((liquid, gas), ports, strict=True)]
    combined = ProcessContribution(
        port_rates_mol_s={k: v for c in contributions for k, v in c.port_rates_mol_s.items()},
        port_enthalpies_J_mol={k: v for c in contributions
                              for k, v in c.port_enthalpies_J_mol.items()})
    system = CoupledSystem(receiver, (CoupledProcess("flash-outlets", lambda _: combined),),
        ports=ports, numerical_form=numerical_form,
        policy=ODESolverPolicy("flash-feed", method="BDF", rtol=1e-11, atol=1e-13))
    initial = CoupledState({"fluid": 2}, 600, 2000)
    report = system.advance(initial, 5)
    final = report.final_state
    assert final.amounts_mol["fluid"] == pytest.approx(2.05)
    assert sum(final.exchanged_mol.values()) == pytest.approx(.05)
    expected_energy = 5*stage_result.outlet.enthalpy_rate_W
    assert sum(final.exchange_energy_J.values()) == pytest.approx(expected_energy)
    assert receiver.internal_energy_j(final)-receiver.internal_energy_j(initial) == pytest.approx(
        expected_energy, abs=1e-5)
    assert report.energy_balance_residual_J == pytest.approx(0, abs=1e-5)


def test_pressure_stage_rejects_inconsistent_reference_and_operating_direction():
    device, upstream = stage("compressor")
    with pytest.raises(ValueError, match="EOS/reference"):
        device.evaluate(HelmholtzStream(replace(upstream, enthalpy_J_mol=99999), .01))
    with pytest.raises(ValueError, match="direction"):
        replace(device, outlet_pressure_Pa=1e5).evaluate(HelmholtzStream(upstream, .01))
    with pytest.raises(ValueError, match="efficiency"):
        replace(device, isentropic_efficiency=1.1)
    with pytest.raises(ValueError, match="two-phase"):
        HelmholtzStream(upstream, .01).separate()
