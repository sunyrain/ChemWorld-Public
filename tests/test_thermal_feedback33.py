"""Real finite-heater feedback, original ownership and no-controller compatibility."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.thermal_feedback import thermal_feedback_world
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    SpeciesInternalEnergy,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.thermal_feedback import (
    ThermalFeedbackSpec,
    augment_thermal_feedback,
    feedback_key,
    thermal_feedback_point,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.thermal_feedback import with_thermal_feedback


def point(*, device=None, spec=None, **values):
    fields = {"sample_temperature_K": 300.0, "integral_K": 0.0, "filtered_temperature_K": 300.0}
    fields.update(values)
    return thermal_feedback_point(
        spec or ThermalFeedbackSpec(proportional_gain=2.0),
        device or ThermalDeviceSpec(maximum_power_W=5.0, maximum_cooling_power_W=3.0),
        **fields,
    )


def test_antiwindup_tracks_real_power_even_when_command_is_within_range():
    p = point()
    assert p["command_limited"] is True
    assert p["power_limited"] is True
    assert p["actual_heater_power_W"] == 5.0
    assert p["realized_command_K"] == 325.0
    assert p["integral_rate_K_s"] == pytest.approx(0.1 * 50 + 0.5 * (325 - 450))
    p = point(spec=ThermalFeedbackSpec(proportional_gain=1.0))
    assert not p["command_limited"]
    assert p["power_limited"]
    assert p["integral_rate_K_s"] < 0


def test_cooling_power_is_actual_signed_heat():
    p = point(sample_temperature_K=400.0, filtered_temperature_K=400.0)
    assert p["actual_heater_power_W"] == -3.0
    assert p["realized_command_K"] == 385.0
    assert p["integral_rate_K_s"] > 0


def test_disabled_heater_freezes_integral_but_tracks_real_measurement():
    p = point(device=ThermalDeviceSpec(enabled=False), integral_K=-20, filtered_temperature_K=290.0)
    assert p["actual_heater_power_W"] == p["integral_rate_K_s"] == 0
    assert p["filter_rate_K_s"] == 20.0


def test_derivative_uses_filtered_measurement_not_setpoint_derivative():
    spec = ThermalFeedbackSpec(derivative_gain_s=2.0)
    a = point(spec=spec, filtered_temperature_K=299.0)
    b = point(spec=spec, device=ThermalDeviceSpec(setpoint_K=360.0), filtered_temperature_K=299.0)
    assert a["filter_rate_K_s"] == b["filter_rate_K_s"]
    assert b["raw_command_K"] - a["raw_command_K"] == 10.0


@pytest.mark.parametrize("kind", ["P", "PI", "PD", "PID"])
def test_named_types_are_explicit_gain_specializations(kind):
    spec = ThermalFeedbackSpec(
        integral_gain_s_inv=0.1 if "I" in kind else 0,
        derivative_gain_s=0.5 if "D" in kind else 0,
        antiwindup_rate_s_inv=0.2 if "I" in kind else 0,
    )
    assert spec.controller_type == kind
    assert ThermalFeedbackSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    if "I" not in kind:
        assert point(spec=spec)["integral_rate_K_s"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"proportional_gain": -1},
        {"integral_gain_s_inv": float("nan")},
        {"derivative_filter_time_s": 0},
        {"minimum_command_K": 440},
        {"integral_gain_s_inv": 0},
        {"pause_rule": "reset"},
    ],
)
def test_physical_feedback_spec_rejects_undefined_controls(changes):
    with pytest.raises(ValueError):
        ThermalFeedbackSpec(**changes)


@pytest.mark.parametrize(
    "device",
    [
        ThermalDeviceSpec(mode="isothermal"),
        ThermalDeviceSpec(conductance_W_K=0),
    ],
)
def test_controller_requires_real_finite_conductance_boundary(device):
    with pytest.raises(ValueError):
        point(device=device)


def test_original_system_owns_energy_and_no_feedback_inventory_is_added():
    thermo = ConstantCvThermodynamics({"A": SpeciesInternalEnergy(0, 20)}, 20)
    base = CoupledSystem(thermo, heat_flux=lambda _: HeatFlux(999, 2, 3))
    initial = CoupledState({"A": 0.01}, 300, 1)
    compiled, state = augment_thermal_feedback(
        base,
        initial,
        ThermalFeedbackSpec(),
        ThermalDeviceSpec(),
        "heater",
        integral_K=0,
        filtered_temperature_K=300,
    )
    assert compiled.thermodynamics is thermo
    assert state.amounts_mol == initial.amounts_mol
    assert compiled.heat_flux(state) == HeatFlux(5, 2, 3)
    assert set(state.coordinates) == {
        feedback_key("heater", "integral_K"),
        feedback_key("heater", "filtered_temperature_K"),
    }


def test_uninstalled_helper_returns_the_same_original_objects():
    composition = thermal_feedback_world()
    original = composition.physics.initial_states["reactor"]
    records = dict(original.equipment.equipment)
    record = records["feedback-heater"]
    records["feedback-heater"] = replace(
        record, settings={k: v for k, v in record.settings.items() if k != "thermal_feedback"}
    )
    original = original.replace(
        equipment=replace(original.equipment, equipment=records),
        state_slots=replace(original.state_slots, slots=()),
    )
    system = CoupledSystem(ConstantCvThermodynamics({"A": SpeciesInternalEnergy(0, 20)}))
    initial = CoupledState({"A": 0.01}, 300, 1)
    actual, packed = with_thermal_feedback(original, system, initial)
    assert actual is system
    assert packed is initial


def test_real_runtime_records_original_heat_and_exact_memory_continuation():
    runtime = ChemWorldRuntime(composition=thermal_feedback_world("PID"))
    replay = None
    try:
        runtime.reset(seed=33)
        result = runtime.execute(
            {"operation": "heat", "target_temperature_K": 350, "duration_s": 1}
        )
        assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.temperature_K > 300
        assert 0 < runtime._state.ledger.energy_jacket_J <= 30 + 1e-7
        payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
        replay = ChemWorldRuntime.from_snapshot(payload)
        for session in (runtime, replay):
            result = session.execute({"operation": "wait", "duration_s": 1})
            assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.to_dict() == replay._state.to_dict()
    finally:
        runtime.close()
        if replay:
            replay.close()
