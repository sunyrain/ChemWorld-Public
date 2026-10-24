"""The complete packed residual differentiates its actual temperature view."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from chemworld.physchem import coupled_residual
from chemworld.physchem.battery_models import battery_system
from chemworld.physchem.battery_spatial_thermal import spatial_thermal_system
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import ResidualEvaluator, finite_jacobian
from test_battery_hysteresis_analysis import model as ocp_model
from test_battery_spatial_thermal import model as thermal_model


class PackingCaptured(BaseException):
    """Stop after real advance packing, without invoking numerical integration."""


def dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def capture(monkeypatch, physical, state):
    packed = {}
    physical_derivatives = []
    original = physical.residual_rhs_jacobian

    def observe(view, identities):
        jacobian = original(view, identities)
        physical_derivatives.append((jacobian, dense(jacobian).copy()))
        return jacobian

    if original is not None:
        physical = replace(physical, residual_rhs_jacobian=observe)

    def before_integration(system, initial, **kwargs):
        packed.update(system=system, initial=np.asarray(initial).copy())
        raise PackingCaptured()

    monkeypatch.setattr(coupled_residual, "solve_residual", before_integration)
    with pytest.raises(PackingCaptured):
        physical.advance(state, 1.0)
    evaluator = ResidualEvaluator(packed["system"])
    initial = packed["initial"]
    names = tuple(variable.name for variable in packed["system"].variables)
    return evaluator, initial, names, physical_derivatives


def check_temperature_chain(evaluator, initial, names, physical_derivatives, *, isothermal):
    zero = np.zeros(len(initial))
    temperature = names.index("temperature")
    analytic = dense(evaluator.derivative("state", 0.0, initial, zero))
    raw, before = physical_derivatives[-1]
    # Packing may not mutate or suppress a physical-domain derivative provider.
    np.testing.assert_array_equal(dense(raw), before)
    assert np.max(np.abs(before[:, temperature])) > 1e-6
    others = np.arange(len(initial)) != temperature
    np.testing.assert_array_equal(analytic[:, others], -before[:, others])
    mass = dense(evaluator.derivative("rate", 0.0, initial, zero))
    np.testing.assert_array_equal(mass[temperature], np.eye(len(initial))[temperature])
    for step in (0.01, 0.001):
        plus, minus = initial.copy(), initial.copy()
        plus[temperature] += step
        minus[temperature] -= step
        measured = (
            evaluator.value(0.0, plus, zero) - evaluator.value(0.0, minus, zero)
        ) / (2 * step)
        if isothermal:
            np.testing.assert_array_equal(measured, 0)
            np.testing.assert_array_equal(analytic[:, temperature], 0)
            np.testing.assert_array_equal(analytic[temperature], 0)
        else:
            np.testing.assert_array_equal(analytic[:, temperature], -before[:, temperature])
            assert np.max(np.abs(measured)) > 1e-6
            np.testing.assert_allclose(analytic[:, temperature], measured, rtol=2e-6, atol=2e-8)
    return analytic


@pytest.mark.parametrize("family", ("spm", "mpm"))
def test_real_ocp_battery_packed_temperature_column_matches_its_fixed_view(monkeypatch, family):
    spec = ocp_model(family, "current_sigmoid")
    state = spec.initial_state()
    program = ScalarControlProgram(((0.0, 0.2),), "A")
    physical = battery_system(spec, program)
    assert physical.isothermal
    captured = capture(monkeypatch, physical, state)
    check_temperature_chain(*captured, isothermal=True)
    assert state == spec.initial_state()


def test_real_spatial_thermal_packing_keeps_dynamic_temperature_feedback(monkeypatch):
    spec = thermal_model()
    state = spec.initial_state()
    program = ScalarControlProgram(((0.0, 0.2),), "A")
    physical = spatial_thermal_system(spec, program)
    assert not physical.isothermal
    captured = capture(monkeypatch, physical, state)
    check_temperature_chain(*captured, isothermal=False)
    assert state == spec.initial_state()


def reaction(isothermal, representation):
    def rates(view):
        rate = 0.4 * view.temperature_K / 300 * view.amounts_mol["A"]
        return ProcessContribution({"A": -rate, "B": rate})

    def derivative(view, identities):
        jacobian = np.zeros((len(identities), len(identities)))
        a, b, t = (identities.index(name) for name in ("A", "B", "temperature"))
        jacobian[a, a] = -0.4 * view.temperature_K / 300
        jacobian[b, a] = -jacobian[a, a]
        jacobian[a, t] = -0.4 / 300 * view.amounts_mol["A"]
        jacobian[b, t] = -jacobian[a, t]
        if representation == "sparse":
            return sparse.csc_matrix(jacobian)
        # A consumer must also accept an immutable domain-owned dense result.
        jacobian.setflags(write=False)
        return jacobian

    return CoupledSystem(
        ConstantCvThermodynamics({name: SpeciesInternalEnergy(0, 0) for name in ("A", "B")}, 1),
        (CoupledProcess("conversion", rates),),
        isothermal=isothermal,
        numerical_form="residual",
        residual_rhs_jacobian=derivative,
    )


@pytest.mark.parametrize("isothermal", (True, False))
@pytest.mark.parametrize("representation", ("dense", "sparse"))
def test_jacobian_protocol_preserves_other_columns_and_provider_storage(
    monkeypatch, isothermal, representation
):
    physical = reaction(isothermal, representation)
    state = CoupledState({"A": 1.0, "B": 0.5}, 900.0, 1.0)
    evaluator, initial, names, recorded = capture(monkeypatch, physical, state)
    analytic = check_temperature_chain(
        evaluator, initial, names, recorded, isothermal=isothermal
    )
    measured = dense(finite_jacobian(
        lambda y: evaluator.value(0.0, y, np.zeros(len(y))), initial, np.ones(len(initial))
    ))
    np.testing.assert_allclose(analytic, measured, rtol=2e-8, atol=2e-10)


def test_numerical_derivative_without_provider_already_obeys_fixed_temperature(monkeypatch):
    physical = replace(reaction(True, "dense"), residual_rhs_jacobian=None)
    state = CoupledState({"A": 1.0, "B": 0.5}, 900.0, 1.0)
    evaluator, initial, names, recorded = capture(monkeypatch, physical, state)
    analytic = dense(evaluator.derivative("state", 0.0, initial, np.zeros(len(initial))))
    assert not recorded
    np.testing.assert_array_equal(analytic[:, names.index("temperature")], 0)
