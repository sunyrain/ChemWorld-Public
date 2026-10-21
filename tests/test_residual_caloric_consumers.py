"""Native caloric root/derivative consumers; part of 21B functional coverage."""

import json

import numpy as np
import pytest

from chemworld.physchem.enthalpy import caloric_residual_system, caloric_temperature
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualNumericalError,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)
from chemworld.physchem.transport_devices import FiniteTransportSpec
from chemworld.runtime import ChemWorldRuntime, SessionSnapshot
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec


@pytest.mark.parametrize("basis", ["enthalpy", "internal_energy"])
def test_caloric_temperature_and_native_energy_parameter_responses(basis):
    def energy(t, p):
        return p[0] * (t - 300) + 0.05 * (t**2 - 300**2) + p[1]

    values = (20, -50)
    system = caloric_residual_system(
        energy,
        lambda t, p: p[0] + 0.1 * t,
        energy(330, values),
        (280, 400),
        energy_basis=basis,
        parameters=(
            ResidualVariable("capacity", "J/K", 20, "parameter", lower=0),
            ResidualVariable("reference", "J", 100, "parameter"),
        ),
        parameter_values=values,
        energy_parameter_jacobian=lambda t, p: [t - 300, 1],
    )
    result = solve_steady_residual(
        system, [290], sensitivity=True, policy=RootPolicy(tolerance=1e-12)
    )
    assert result.root.success, result
    assert result.root.value == pytest.approx([330], abs=1e-9)
    assert result.parameter_response[0] == pytest.approx([1 / 53, -30 / 53, -1 / 53])
    assert system.equations[0].unit == "J"
    assert [p.unit for p in system.parameters] == ["J", "J/K", "J"]


@pytest.mark.parametrize("temperature", [280, 310, 400])
def test_scalar_residual_caloric_closure_preserves_temperature_tolerances(temperature):
    def energy(t):
        return 50 * (t - 300) + 0.5 * (t - 300) ** 2 + 7e6

    result = caloric_temperature(
        energy,
        energy(temperature),
        (280, 400),
        energy_basis="enthalpy",
        algorithm="residual",
        heat_capacity_at_temperature=lambda t: t - 250,
    )
    assert result == pytest.approx(temperature, abs=1e-9)


def test_missing_physical_parameter_derivative_is_not_fabricated():
    system = caloric_residual_system(
        lambda t, p: p[0] * (t - 300),
        lambda t, p: p[0],
        600,
        (280, 400),
        energy_basis="enthalpy",
        parameters=(ResidualVariable("capacity", "J/K", 20, "parameter"),),
        parameter_values=(20,),
    )
    assert solve_steady_residual(system, [300]).root.success
    with pytest.raises(ValueError, match="parameter derivative is unavailable"):
        solve_steady_residual(system, [300], sensitivity=True)


@pytest.mark.parametrize("capacity", [None, lambda t: -1, lambda t: np.nan])
def test_residual_caloric_algorithm_never_falls_back_when_derivative_is_unavailable(capacity):
    with pytest.raises((ValueError, ResidualDomainError), match="heat capacity"):
        caloric_temperature(
            lambda t: 20 * (t - 300),
            600,
            (280, 400),
            energy_basis="enthalpy",
            algorithm="residual",
            heat_capacity_at_temperature=capacity,
        )


@pytest.mark.parametrize("algorithm", ["residual", "brentq"])
def test_transport_caloric_mode_roundtrips_and_both_algorithms_consume_the_same_datum(algorithm):
    original = FiniteTransportSpec(synthetic_transport_properties(), caloric_algorithm=algorithm)
    spec = FiniteTransportSpec.from_dict(json.loads(json.dumps(original.to_dict(), sort_keys=True)))
    assert spec.caloric_algorithm == algorithm
    amounts = {"water": 0.8, "ethanol": 0.2}
    assert spec.properties.heat_capacity(amounts, 320) == pytest.approx(82)
    assert spec.at_ph(0, amounts, spec.enthalpy(0, amounts, 320)) == pytest.approx(320, abs=1e-9)
    assert spec.at_ph(0, dict.fromkeys(amounts, 0), spec.capacities_j_k[0] * (320 - 298.15)) == (
        pytest.approx(320, abs=1e-9)
    )


def test_transport_residual_failure_rolls_back_real_material_clock_and_pump_work(monkeypatch):
    import chemworld.physchem.enthalpy as module

    session = ChemWorldRuntime(composition=transport_world_spec())
    try:
        session.reset(seed=24)
        assert (
            session.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 5}).info[
                "transaction_status"
            ]
            == "committed"
        )
        before = session._state

        def fail(*args, **kwargs):
            raise ResidualNumericalError("injected caloric residual failure")

        monkeypatch.setattr(module, "solve_steady_residual", fail)
        result = session.execute({"operation": "wait", "duration_s": 3})
        assert result.info["transaction_status"] != "committed"
        assert result.info["operation_outcome"] == "numerical_failure"
        assert session._state.species_amounts == before.species_amounts
        assert session._state.inactive_vessels == before.inactive_vessels
        assert session._state.state_slots == before.state_slots
        assert session._state.ledger.time_s == before.ledger.time_s
        assert session._state.ledger.flow_pump_work_J == before.ledger.flow_pump_work_J
    finally:
        session.close()


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_transport_residual_snapshot_resume_preserves_reading_and_exact_resources(noise):
    original = ChemWorldRuntime(composition=transport_world_spec(), observation_noise_mode=noise)
    restored = None
    try:
        original.reset(seed=24)
        for action in (
            {"operation": "set_flow_rate", "flow_rate_mL_min": 5},
            {"operation": "wait", "duration_s": 3},
        ):
            assert original.execute(action).info["transaction_status"] == "committed"
        restored = ChemWorldRuntime.from_snapshot(
            SessionSnapshot.from_dict(
                json.loads(json.dumps(original.snapshot().to_dict(), sort_keys=True))
            )
        )
        for action in (
            {"operation": "wait", "duration_s": 4},
            {"operation": "measure", "instrument": "flow_meter"},
        ):
            a, b = original.execute(action), restored.execute(action)
            assert a.info["transaction_status"] == b.info["transaction_status"] == "committed"
            assert a.observation.to_dict() == b.observation.to_dict()
        assert original._state.ledger == restored._state.ledger
        assert original._state.state_slots == restored._state.state_slots
        assert a.observation.channel("flow_rate").unit == "mL/min"
    finally:
        original.close()
        if restored is not None:
            restored.close()
