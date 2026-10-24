"""Real finite heat-recovery behavior and input/ownership refusal."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.finite_heat_recovery import (
    HeatRecoverySolveError,
    advance_finite_heat_recovery,
    heat_recovery_inventory,
    heat_recovery_profile,
)
from chemworld.physchem.finite_heat_recovery_specs import FiniteHeatRecoverySpec
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.finite_heat_recovery import (
    FiniteHeatRecoveryAdapter,
    heat_recovery_record,
)
from chemworld.world.finite_heat_recovery_world import finite_heat_recovery_world_spec
from reference.finite_heat_recovery31_cases import (
    fixed_cases,
    initial_case,
    reference,
    setup,
    spec_case,
)


@pytest.mark.parametrize("case_index", [0, 5, 12, 17, 18, 23])
def test_real_finite_exchange_matches_independent_reference(case_index):
    _, spec, composition, reverse, flows, duration, variable = fixed_cases()[case_index]
    amounts, temperatures = setup(spec, composition, reverse)
    initial = initial_case(spec, composition, reverse)
    expected = reference(spec, amounts, temperatures, flows, duration, variable=variable)
    result, diagnostic = advance_finite_heat_recovery(spec, initial, flows, duration)
    for actual, target in zip(result.final_state.amounts_mol, expected["amounts"], strict=True):
        assert actual == pytest.approx(target, rel=3e-7, abs=1e-10)
    assert result.final_state.enthalpies_J == pytest.approx(
        expected["enthalpies_J"], rel=3e-7, abs=1e-7
    )
    assert diagnostic["heat_left_to_right_W"] == pytest.approx(
        expected["duty_left_to_right_W"], rel=3e-7
    )
    assert result.enthalpy_residual_J == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("index", range(24, 32))
def test_each_finite_boundary_and_zero_flow_is_real(index):
    _, spec, composition, reverse, flows, duration, variable = fixed_cases()[index]
    amounts, temperatures = setup(spec, composition, reverse)
    expected = reference(spec, amounts, temperatures, flows, duration, variable=variable)
    result, diagnostic = advance_finite_heat_recovery(spec, initial_case(spec), flows, duration)
    assert result.actual_duration_s == pytest.approx(expected["actual_duration_s"], abs=1e-6)
    assert result.stopping_event == expected["stop"]
    if not all(flows):
        assert diagnostic["heat_left_to_right_W"] == 0
    assert diagnostic["shaft_work_J"] == pytest.approx(expected["shaft_work_J"], abs=1e-7)


def test_zero_conductance_and_equal_inlet_temperatures():
    spec = spec_case(ua=0)
    initial = initial_case(spec)
    assert heat_recovery_profile(spec, initial, (8 / 6e7, 12 / 6e7))["heat_left_to_right_W"] == 0
    # Identical sources and pumps have equal exchanger inlets even while flowing.
    amounts, _ = setup(spec)
    rows = (amounts[0], amounts[1], dict(amounts[0]), amounts[3])
    equal = heat_recovery_inventory(spec_case(), rows, (310.0, 300.0, 310.0, 300.0))
    assert (
        heat_recovery_profile(spec_case(), equal, (8 / 6e7, 8 / 6e7))["heat_left_to_right_W"] == 0
    )
    result, diagnostic = advance_finite_heat_recovery(spec_case(), equal, (8 / 6e7, 8 / 6e7), 5)
    assert diagnostic["outlet_temperatures_K"][0] == diagnostic["outlet_temperatures_K"][1]
    assert result.final_state.amounts_mol[1]["water"] > equal.amounts_mol[1]["water"]
    assert result.final_state.amounts_mol[3]["water"] > equal.amounts_mol[3]["water"]


def test_roundtrip_does_not_share_mutable_property_or_pump_inputs():
    spec = spec_case(variable=True)
    data = json.loads(json.dumps(spec.to_dict()))
    restored = FiniteHeatRecoverySpec.from_dict(data)
    assert json.loads(json.dumps(restored.to_dict())) == data
    data["properties"]["packages"][0]["correlations"][0]["coefficients"]["a"] = 999
    assert restored.to_dict() != data


def test_root_budget_failure_is_explicit_and_input_unchanged():
    spec = replace(spec_case(variable=True), maximum_root_iterations=1)
    initial = initial_case(spec)
    before = initial.amounts_mol, initial.enthalpies_J
    with pytest.raises(HeatRecoverySolveError, match="root failed"):
        heat_recovery_profile(spec, initial, (8 / 6e7, 12 / 6e7))
    assert (initial.amounts_mol, initial.enthalpies_J) == before


def test_pump_heating_cannot_clip_an_unreachable_temperature():
    spec = spec_case()
    spec = replace(
        spec,
        pumps=(PumpSpec(1e12, 1e9, 0.7), spec.pumps[1]),
        pipes=(PipeSpec(1, 0.00001), spec.pipes[1]),
    )
    with pytest.raises(ValueError, match="outside"):
        heat_recovery_profile(spec, initial_case(spec), (8 / 6e7, 12 / 6e7))


@pytest.mark.parametrize("shared", [False, True])
def test_original_runtime_independent_controls_and_exact_snapshot(shared):
    spec = spec_case()
    amounts, temperatures = setup(spec)
    world = finite_heat_recovery_world_spec(spec, amounts, temperatures, shared_clock=shared)
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=31)
    for side, flow in (("left", 8), ("right", 12)):
        result = runtime.execute(
            {"operation": "set_flow_rate", "flow_path": side, "flow_rate_mL_min": flow}
        )
        assert result.info["transaction_status"] == "committed", result.info
    assert heat_recovery_record(runtime._state).settings["flow_controls"] == {
        "left": 8 / 6e7,
        "right": 12 / 6e7,
    }
    action = {"operation": "wait" if shared else "run_flow", "duration_s": 5}
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    runtime.execute(action)
    restored.execute(action)
    assert runtime.snapshot().to_dict() == restored.snapshot().to_dict()
    assert runtime._state.ledger.energy_jacket_J == 0
    assert runtime._state.ledger.flow_heat_J == 0
    if shared:
        assert set(runtime._state.vessel_elapsed_s.values()) == {10}


@pytest.mark.parametrize("path", [None, "process", "rinse", "unknown"])
def test_flow_side_is_explicit_and_heat_control_rejects(path):
    spec = spec_case()
    amounts, temperatures = setup(spec)
    world = finite_heat_recovery_world_spec(spec, amounts, temperatures)
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset()
    adapter = FiniteHeatRecoveryAdapter(world)
    before = runtime._state.to_dict()
    with pytest.raises(ValueError, match="left or right"):
        adapter.configure_flow(runtime._state, {"flow_path": path, "flow_rate_mL_min": 1})
    refused = adapter.advance(runtime._state, {"duration_s": 1}, heat=True)
    assert refused.outcome == "rejected" and refused.state is None
    assert runtime._state.to_dict() == before


def test_initial_receiver_capacity_is_zero_time_stop():
    spec = replace(spec_case(), receiver_capacities_L=(0.002, 1.0))
    result, diagnostic = advance_finite_heat_recovery(
        spec, initial_case(spec), (8 / 6e7, 12 / 6e7), 10
    )
    assert result.stopping_event == "left_receiver_full"
    assert result.actual_duration_s == 0 and diagnostic["shaft_work_J"] == 0


@pytest.mark.parametrize("change", ["ion", "nonideal", "conductance", "duplicate", "body"])
def test_unsupported_or_invalid_model_declarations_refuse(change):
    spec = spec_case()
    with pytest.raises(ValueError):
        if change == "ion":
            packages = list(spec.properties.packages)
            packages[0] = replace(packages[0], component=replace(packages[0].component, charge=1))
            replace(spec, properties=replace(spec.properties, packages=tuple(packages)))
        elif change == "nonideal":
            replace(spec, properties=replace(spec.properties, volume_mixing_method="unknown"))
        elif change == "conductance":
            replace(spec, conductance_W_K=-1)
        elif change == "duplicate":
            replace(spec, left_feed_vessel_id=spec.right_feed_vessel_id)
        else:
            replace(spec, capacities_J_K=(0, 7, 6, 8))


def test_extra_owned_state_refuses_without_discarding_it():
    from chemworld.foundation.state_ownership import StateFieldSpec, StateSlot, StateSlotSpec
    from chemworld.runtime.finite_heat_recovery import unpack_heat_recovery

    spec = spec_case()
    amounts, temperatures = setup(spec)
    runtime = ChemWorldRuntime(
        composition=finite_heat_recovery_world_spec(spec, amounts, temperatures)
    )
    runtime.reset()
    extra = StateSlot(
        StateSlotSpec(
            "extra", "equipment", "heat_recovery", "foreign", (StateFieldSpec("x", "mol"),)
        ),
        "reactor",
        {"x": 1.0},
    )
    changed = runtime._state.replace(
        state_slots=replace(
            runtime._state.state_slots, slots=(*runtime._state.state_slots.slots, extra)
        )
    )
    with pytest.raises(ValueError, match="state slots"):
        unpack_heat_recovery(changed)
    assert changed.state_slots.slots[-1] == extra
