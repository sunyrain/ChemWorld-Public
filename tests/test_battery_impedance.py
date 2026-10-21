"""Actual native SPM/SPMe/DFN consumers of the common frequency response."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import BatterySPMSpec, BatteryThermalSpec, F, ParticleSpec
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEvaluator,
    finite_jacobian,
)


def model(family):
    particles = {"negative": ParticleSpec(cells=8), "positive": ParticleSpec(cells=8)}
    electrolyte = BatteryElectrolyteSpec(cells=(3, 2, 3))
    if family == "dfn":
        return BatteryDFNSpec(**particles, electrolyte=electrolyte)
    return BatterySPMSpec(**particles, electrolyte=electrolyte if family == "spme" else None)


@pytest.mark.parametrize("family", ["spm", "spme", "dfn"])
def test_actual_native_residual_conserves_inventory_and_charge_in_frequency_domain(family):
    spec = model(family)
    state = spec.initial_state()
    original = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    system, initial = battery_impedance_system(spec, state)
    result = battery_impedance(spec, [0.001, 0.01, 0.1], state=state)
    assert result.success
    indices = {v.name: i for i, v in enumerate(system.variables)}
    for point, z in zip(result.response.points, result.impedance_ohm, strict=True):
        assert z.real > 0 and z.imag < 0
        y = point.response[:, 0]
        assert y[indices["battery:current_A"]] == pytest.approx(1)
        negative = [indices[k] for k in state.amounts_mol if k.startswith("negative:")]
        positive = [indices[k] for k in state.amounts_mol if k.startswith("positive:")]
        assert sum(y[negative]) + sum(y[positive]) == pytest.approx(0, abs=1e-12)
        assert F * sum(y[negative]) + y[indices["discharged_charge_C"]] == pytest.approx(
            0, abs=1e-8
        )
        assert F * sum(y[positive]) - y[indices["discharged_charge_C"]] == pytest.approx(
            0, abs=1e-8
        )
        assert y[indices["discharged_charge_C"]] == pytest.approx(
            1 / (2j * np.pi * point.frequency_Hz)
        )
        if spec.electrolyte is not None:
            assert sum(y[indices[k]] for k in spec.electrolyte.keys) == pytest.approx(0, abs=1e-12)
    assert original == (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    assert not initial.flags.writeable
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("family", ["spm", "spme", "dfn"])
def test_series_resistance_adds_real_port_impedance_not_imaginary(family):
    spec = model(family)
    a = battery_impedance(spec, [0.003, 0.02])
    b = battery_impedance(replace(spec, series_resistance_ohm=0.13), [0.003, 0.02])
    np.testing.assert_allclose(
        np.array(b.impedance_ohm) - np.array(a.impedance_ohm), 0.13, rtol=1e-7, atol=1e-9
    )


def test_dfn_reuses_analytic_state_derivative_and_matches_independent_differences():
    spec = model("dfn")
    system, y = battery_impedance_system(spec, spec.initial_state())
    evaluator = ResidualEvaluator(system)
    analytic = evaluator.derivative("state", 0, y, np.zeros(len(y))).toarray()
    numerical = finite_jacobian(
        lambda value: evaluator.value(0, value, np.zeros(len(y))), y, system.scales
    )
    np.testing.assert_allclose(
        analytic * system.scales, numerical * system.scales, rtol=1e-5, atol=1e-7
    )
    result = battery_impedance(spec, [0.01])
    assert result.response.diagnostics["derivative_modes"]["state"] == "analytic"
    assert result.response.diagnostics["derivative_modes"]["parameter"] == "analytic"


def test_nonstationary_inventory_is_not_automatically_relaxed_or_clipped():
    spec = model("spm")
    state = spec.initial_state()
    values = dict(state.amounts_mol)
    keys = spec.negative.keys("negative")
    delta = values[keys[0]] * 0.1
    values[keys[0]] -= delta
    values[keys[1]] += delta
    altered = replace(state, amounts_mol=values)
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        battery_impedance(spec, [0.01], state=altered)
    assert dict(altered.amounts_mol) == values


def test_cutoff_thermal_and_wrong_state_reject():
    spec = model("spm")
    with pytest.raises(ResidualDomainError, match="stopping boundary"):
        battery_impedance(replace(spec, minimum_voltage_V=3.84), [0.01])
    with pytest.raises(ValueError, match="isothermal"):
        battery_impedance(replace(spec, thermal=BatteryThermalSpec()), [0.01])
    with pytest.raises(ValueError, match="inventory and coordinates"):
        battery_impedance(spec, [0.01], state=replace(spec.initial_state(), coordinates={}))


def test_qualified_default_dfn_mesh_and_existing_ode_spm_inputs():
    result = battery_impedance(BatteryDFNSpec(), [0.001, 0.1])
    assert result.success
    assert len(result.response.variables) > 3000
    spec = model("spm")
    first = battery_impedance(spec, [0.01])
    second = battery_impedance(replace(spec, numerical_form="ode"), [0.01])
    np.testing.assert_allclose(first.impedance_ohm, second.impedance_ohm, rtol=1e-12)
