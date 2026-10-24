"""Read-only composite analysis retains phase identity and old analysis boundaries."""

import json
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.examples.battery_composite import default_spec
from chemworld.physchem.battery_composite import BatteryCompositeSPMSpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import battery_spec_from_dict
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec


def equilibrium_spec():
    # Authored equilibrium, not a repair of a supplied state: both negative OCPs
    # are 0.13 V, while the original demonstration retains its 12.5 mV mismatch.
    spec = default_spec(cells=4)
    alpha, beta = spec.negative_phases
    return replace(spec, negative_phases=(alpha, replace(beta, initial_fraction=0.6)))


def snapshot(state):
    return {item.name: dict(value) if isinstance(value, Mapping) else value
            for item in fields(state) for value in (getattr(state, item.name),)}


def test_codec_roundtrip_keeps_independent_phase_inputs_and_full_response_names():
    spec = equilibrium_spec()
    payload = json.loads(json.dumps(spec.to_dict()))
    restored = battery_spec_from_dict(payload)
    assert isinstance(restored, BatteryCompositeSPMSpec)
    assert restored == spec and restored.to_dict() == payload
    assert "numerical_form" not in payload
    assert restored.numerical_form == "residual"
    assert not hasattr(restored, "negative") and not hasattr(restored, "positive")
    assert [p.phase_id for p in restored.phases] == ["alpha", "beta", "positive"]
    for old, new in zip(spec.phases, restored.phases, strict=True):
        assert new.source == old.source and new.ocv == old.ocv and new.particle == old.particle

    before = snapshot(restored.initial_state())
    first = battery_impedance(spec, [0.003, 0.2])
    second = battery_impedance(restored, [0.003, 0.2])
    assert first.success and second.success
    assert first.response.to_dict() == second.response.to_dict()
    assert first.to_dict() == second.to_dict()
    names = {v.name for v in second.response.variables}
    assert set(restored.inventory_keys) <= names
    assert set(restored.coordinate_keys) <= names
    assert set(restored.algebraic_names) <= names
    assert len(names) == len(restored.inventory_keys) + 2 + len(restored.algebraic_names)
    assert snapshot(restored.initial_state()) == before


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_analysis_of_actual_world_slot_does_not_change_snapshot_or_clock(noise, request):
    runtime = ChemWorldRuntime(composition=battery_world_spec(equilibrium_spec()),
                               observation_noise_mode=noise)
    request.addfinalizer(runtime.close)
    runtime.reset(seed=17)
    original_raw = runtime.snapshot().to_dict()
    original = json.loads(json.dumps(original_raw))
    _, slot, spec, state, _ = unpack_battery(runtime._state)
    before = snapshot(state)
    system, initial = battery_impedance_system(spec, state)
    result = battery_impedance(spec, [0.001, 0.03, 2.0], state=state)
    assert result.success
    assert not initial.flags.writeable
    assert all(not p.response.flags.writeable for p in result.response.points)
    assert snapshot(state) == before
    assert runtime.snapshot().to_dict() == original_raw
    assert len(system.variables) == len(result.response.variables)
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    restored = ChemWorldRuntime.from_snapshot(original)
    request.addfinalizer(restored.close)
    _, restored_slot, restored_spec, restored_state, _ = unpack_battery(restored._state)
    assert restored_slot == slot
    again = battery_impedance(restored_spec, [0.001, 0.03, 2.0], state=restored_state)
    assert again.response.to_dict() == result.response.to_dict()
    assert json.loads(json.dumps(restored.snapshot().to_dict())) == original


@pytest.mark.parametrize("bad", [
    "missing_phase_shell", "extra_stock", "missing_charge", "extra_coordinate",
    "bad_faraday", "negative_inventory", "nonfinite_inventory", "wrong_temperature",
])
def test_analysis_never_repairs_invalid_inventory_or_state_layout(bad):
    spec = equilibrium_spec()
    state = spec.initial_state()
    amounts, coordinates = dict(state.amounts_mol), dict(state.coordinates)
    key = spec.inventory_keys[0]
    if bad == "missing_phase_shell":
        amounts.pop(key)
    elif bad == "extra_stock":
        amounts["negative:Li"] = 0.0
    elif bad == "missing_charge":
        coordinates.pop("discharged_charge_C")
    elif bad == "extra_coordinate":
        coordinates["battery:negative:hysteresis"] = 0.0
    elif bad == "bad_faraday":
        coordinates["discharged_charge_C"] = 1.0
    elif bad == "negative_inventory":
        amounts[key] = -1e-6
    elif bad == "nonfinite_inventory":
        amounts[key] = float("inf")
    else:
        state = replace(state, temperature_K=state.temperature_K + 1)
    if bad in {"negative_inventory", "nonfinite_inventory"}:
        # Persistent inventory refuses these values before analysis is reached.
        before = snapshot(state)
        with pytest.raises(ValueError, match=r"negative|finite"):
            replace(state, amounts_mol=amounts, coordinates=coordinates)
        assert snapshot(state) == before
        return
    state = replace(state, amounts_mol=amounts, coordinates=coordinates)
    before = snapshot(state)
    with pytest.raises(ValueError):
        battery_impedance(spec, [0.01], state=state)
    assert snapshot(state) == before


@pytest.mark.parametrize("frequencies", [[], [0.0], [-1.0], [np.nan], [np.inf]])
def test_invalid_frequency_request_preserves_every_phase_and_receipt(frequencies):
    spec = equilibrium_spec()
    state = spec.initial_state()
    before = snapshot(state)
    with pytest.raises(ValueError, match="finite positive Hz"):
        battery_impedance(spec, frequencies, state=state)
    assert snapshot(state) == before


@pytest.mark.parametrize("undeclared", ["thermal", "sei", "plating", "negative_ocp"])
def test_eis_support_does_not_admit_undeclared_composite_configuration(undeclared):
    payload = equilibrium_spec().to_dict()
    payload[undeclared] = None
    before = json.loads(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown composite SPM fields"):
        battery_spec_from_dict(payload)
    assert payload == before
