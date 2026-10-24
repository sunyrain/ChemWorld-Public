"""Real finite neutral CSTR, operating limits, calorics and original world ownership."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.reactive_water_treatment import (
    ReactiveWaterTreatmentSpec,
    advance_reactive_water_treatment,
    reactive_water_treatment_inventory,
    reactive_water_treatment_profile,
)
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.reactive_water_treatment import (
    ReactiveWaterTreatmentAdapter,
    reactive_treatment_record,
    unpack_reactive_treatment,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.reactive_water_treatment_world import reactive_water_treatment_world_spec
from reference.reactive_water_treatment49_fixtures import fixed_cases, inventory_case, spec_case
from reference.reactive_water_treatment49_reference import independent


def setup(spec=None, *, shared=False):
    spec = spec or spec_case()
    declaration = reactive_water_treatment_world_spec(
        spec, *inventory_case(spec).amounts_mol, shared_clock=shared
    )
    payload = declaration.to_dict()
    payload["task"]["operations"].append("select_vessel")
    runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
    runtime.reset()
    return runtime


def commit(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("kind", ["forward", "reversible", "nonlinear"])
@pytest.mark.parametrize("flow", [0.0, 10 / 6e7])
def test_cstr_matches_independent_chemistry_and_energy(kind, flow):
    spec = spec_case(kind, 318.15)
    initial = inventory_case(spec)
    reference = independent(spec, initial, flow, 20.0)
    result, diagnostic = advance_reactive_water_treatment(spec, initial, flow, 20.0)
    for actual, expected in zip(result.final_state.amounts_mol, reference["amounts"], strict=True):
        assert list(actual.values()) == pytest.approx(expected, abs=3e-8, rel=3e-7)
    assert result.final_state.enthalpies_J == pytest.approx(reference["enthalpies"], abs=3e-6)
    for key in ("pump_work_J", "hydraulic_work_J", "motor_loss_J", "bath_heat_into_liquids_J"):
        assert diagnostic[key] == pytest.approx(reference[key], abs=3e-6)
    assert diagnostic["enthalpy_residual_J"] == pytest.approx(0, abs=3e-6)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=3e-6)
    assert spec.volume_m3(result.final_state.amounts_mol[1]) * 1000 == pytest.approx(0.005)
    assert (
        result.final_state.amounts_mol[1]["A"] < initial.amounts_mol[1]["A"] if not flow else True
    )


@pytest.mark.parametrize("case", range(12, 16))
def test_actual_pump_pressure_shaft_and_finite_events(case):
    _, spec, flow, duration, _ = fixed_cases()[case]
    initial = inventory_case(spec)
    reference = independent(spec, initial, flow, duration)
    result, diagnostic = advance_reactive_water_treatment(spec, initial, flow, duration)
    assert result.stopping_event == reference["stop"]
    assert result.actual_duration_s == pytest.approx(reference["dt"], abs=2e-7)
    assert diagnostic["throughput_m3"] == pytest.approx(reference["throughput_m3"], abs=1e-12)
    if case < 14:
        assert diagnostic["initial_operating_point"]["actual_flow_m3_s"] < flow


def test_blocked_pump_keeps_batch_reaction_and_zero_work():
    spec = replace(spec_case(), pump=PumpSpec(0.0, 0.0, 0.7))
    initial = inventory_case(spec)
    result, diagnostic = advance_reactive_water_treatment(spec, initial, 10 / 6e7, 20.0)
    assert result.actual_duration_s == 20.0 and result.stopping_event is None
    assert result.final_state.amounts_mol[0] == initial.amounts_mol[0]
    assert result.final_state.amounts_mol[2] == initial.amounts_mol[2]
    assert result.final_state.amounts_mol[1]["A"] < initial.amounts_mol[1]["A"]
    assert diagnostic["pump_work_J"] == diagnostic["throughput_m3"] == 0.0


def test_original_h_is_validated_before_and_integrated_after_projection():
    spec = spec_case()
    initial = inventory_case(spec)
    with pytest.raises(ValueError, match="original treatment H"):
        advance_reactive_water_treatment(
            spec,
            replace(
                initial, enthalpies_J=(initial.enthalpies_J[0] + 1.0, *initial.enthalpies_J[1:])
            ),
            0,
            10,
        )
    result, _ = advance_reactive_water_treatment(spec, initial, 0, 10)
    assert result.final_state.enthalpies_J[1] != initial.enthalpies_J[1]


@pytest.mark.parametrize("kind", ["ion", "density", "volume", "heat", "elevation", "empty_holdup"])
def test_rejects_unsupported_or_inconsistent_physics(kind):
    spec = spec_case()
    with pytest.raises(ValueError):
        if kind in {"ion", "density", "volume"}:
            packages = list(spec.properties.packages)
            package = packages[2]
            if kind == "ion":
                packages[2] = replace(package, component=replace(package.component, charge=1))
            else:
                correlations = list(package.correlations)
                correlations[1] = replace(
                    correlations[1],
                    coefficients={
                        **correlations[1].coefficients,
                        **({"alpha": 0.001} if kind == "density" else {"rho_ref": 1100.0}),
                    },
                )
                packages[2] = replace(package, correlations=tuple(correlations))
            replace(spec, properties=replace(spec.properties, packages=tuple(packages)))
        elif kind == "heat":
            replace(
                spec,
                network=replace(
                    spec.network,
                    reactions=(replace(spec.network.reactions[0], delta_h_J_per_mol=10.0),),
                ),
            )
        elif kind == "elevation":
            replace(spec, pipe=replace(spec.pipe, elevation_change_m=1.0))
        else:
            n = inventory_case(spec).amounts_mol
            reactive_water_treatment_inventory(
                spec, n[0], {k: v / 2 for k, v in n[1].items()}, n[2]
            )


def test_roundtrip_retains_original_network_properties_and_independent_copy():
    spec = spec_case("nonlinear")
    payload = json.loads(json.dumps(spec.to_dict()))
    restored = ReactiveWaterTreatmentSpec.from_dict(payload)
    assert restored.to_dict() == spec.to_dict()
    payload["network"]["reactions"][0]["rate_law"]["parameters"]["k"] = 99
    assert restored.network.reactions[0].rate_law.parameters["k"] == 0.8


def test_reaction_component_in_unreactive_product_has_no_driver():
    spec = spec_case()
    payload = reactive_water_treatment_world_spec(spec, *inventory_case(spec).amounts_mol).to_dict()
    next(c for c in payload["components"] if c["kind"] == "reaction")["vessel"] = "product"
    with pytest.raises(ValueError, match="local driver"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


@pytest.mark.parametrize("shared", [False, True])
def test_real_world_commits_actual_three_tanks_single_energy_and_clock(shared):
    runtime = setup(shared=shared)
    initial = unpack_reactive_treatment(runtime._state)[2]
    commit(runtime, {"operation": "set_flow_rate", "flow_rate_mL_min": 10.0})
    result = commit(runtime, {"operation": "wait" if shared else "run_flow", "duration_s": 20.0})
    assert result.info["operation_outcome"] == "success"
    _, spec, final = unpack_reactive_treatment(runtime._state)
    expected = independent(spec, initial, 10 / 6e7, 20.0)
    for row, n in zip(final.amounts_mol, expected["amounts"], strict=True):
        assert list(row.values()) == pytest.approx(n, abs=3e-8)
    diagnostic = reactive_treatment_record(runtime._state).settings["last_reactive_treatment"]
    assert runtime._state.ledger.flow_pump_work_J == pytest.approx(diagnostic["pump_work_J"])
    assert runtime._state.ledger.heat_loss_J == pytest.approx(
        diagnostic["bath_cooling_J"] + diagnostic["motor_loss_J"]
    )
    assert runtime._state.ledger.time_s == 20.0
    if shared:
        assert set(runtime._state.vessel_elapsed_s.values()) == {20.0}


def test_shared_nonanchor_wait_and_finite_stop():
    runtime = setup()
    commit(runtime, {"operation": "set_flow_rate", "flow_rate_mL_min": 10.0})
    result = commit(runtime, {"operation": "run_flow", "duration_s": 200.0})
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(90.0, abs=2e-7)
    runtime = setup(shared=True)
    commit(runtime, {"operation": "set_flow_rate", "flow_rate_mL_min": 10.0})
    commit(runtime, {"operation": "select_vessel", "vessel": "product"})
    result = commit(runtime, {"operation": "wait", "duration_s": 200.0})
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(90.0, abs=2e-7)


def test_bath_cooling_crossing_stops_real_shared_consumer():
    original = spec_case()
    initial = inventory_case(original)
    first = reactive_water_treatment_profile(original, initial.amounts_mol, 10 / 6e7)
    limit = -first["bath_heat_into_liquids_W"] * 1.5
    spec = replace(original, bath=replace(original.bath, maximum_cooling_power_W=limit))
    result, diagnostic = advance_reactive_water_treatment(
        spec, inventory_case(spec), 10 / 6e7, 20.0
    )
    assert result.stopping_event == "bath_cooling_limit" and 0 < result.actual_duration_s < 20
    assert diagnostic["final_operating_point"]["bath_heat_into_liquids_W"] == pytest.approx(-limit)
    runtime = setup(spec, shared=True)
    commit(runtime, {"operation": "set_flow_rate", "flow_rate_mL_min": 10.0})
    outcome = commit(runtime, {"operation": "wait", "duration_s": 20.0})
    assert outcome.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(result.actual_duration_s, abs=2e-7)


@pytest.mark.parametrize("kind", ["phase", "slot", "H", "model"])
def test_snapshot_rejects_foreign_original_owner_with_anchor_inactive(kind):
    runtime = setup(shared=True)
    commit(runtime, {"operation": "select_vessel", "vessel": "product"})
    payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
    anchor = payload["world_state"]["inactive_vessels"]["reactor"]
    if kind == "phase":
        payload["world_state"]["phases"]["liquid"]["vessel_id"] = "foreign"
    elif kind == "slot":
        anchor["state_slots"]["reactor:material:enthalpy"]["location_id"] = "product"
    elif kind == "H":
        anchor["state_slots"]["reactor:material:enthalpy"]["values"]["enthalpy_J"] += 1.0
    else:
        anchor["equipment"]["treatment"]["settings"]["physical_model"]["pump"][
            "maximum_shaft_power_W"
        ] *= 2
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(payload)


@pytest.mark.parametrize("shared", [False, True])
def test_committed_prefix_json_new_process_continuation_exact(tmp_path, shared):
    runtime = setup(shared=shared)
    commit(runtime, {"operation": "set_flow_rate", "flow_rate_mL_min": 10.0})
    operation = "wait" if shared else "run_flow"
    if shared:
        commit(runtime, {"operation": "select_vessel", "vessel": "product"})
    commit(runtime, {"operation": operation, "duration_s": 5.0})
    source, output = tmp_path / "prefix.json", tmp_path / "continued.json"
    source.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
    commit(runtime, {"operation": operation, "duration_s": 15.0})
    environment = {**os.environ, "PYTHONPATH": str(Path("tests").resolve())}
    child = subprocess.run(
        [
            sys.executable,
            "tests/reference/reactive_water_treatment49_resume.py",
            str(source),
            str(output),
            operation,
            "15",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=50,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        json.dumps(runtime.snapshot().to_dict())
    )


def test_rejects_thermal_control_without_mutating_original_inventory():
    runtime = setup()
    before = copy.deepcopy(runtime._state.to_dict())
    result = ReactiveWaterTreatmentAdapter(runtime.compiled_composition.spec).advance(
        runtime._state, {"operation": "wait", "duration_s": 10}, heat=True
    )
    assert result.outcome == "rejected" and result.state is None
    assert runtime._state.to_dict() == before


def test_initial_full_product_stops_flow_but_zero_flow_still_reacts():
    spec = replace(spec_case(), capacities_L=(0.1, 0.005, 0.001))
    initial = inventory_case(spec)
    flowing, _ = advance_reactive_water_treatment(spec, initial, 10 / 6e7, 10.0)
    assert flowing.stopping_event == "product_full" and flowing.actual_duration_s == 0
    batch, _ = advance_reactive_water_treatment(spec, initial, 0.0, 10.0)
    assert batch.stopping_event is None and batch.actual_duration_s == 10.0
    assert batch.final_state.amounts_mol[1]["A"] < initial.amounts_mol[1]["A"]


@pytest.mark.parametrize("scope", ["material", "vessel"])
def test_shared_partial_lifecycle_cannot_replace_a_coupled_stock(scope):
    runtime = setup(shared=True)
    commit(runtime, {"operation": "wait", "duration_s": 5.0})
    before = runtime._state.to_dict()
    with pytest.raises(ValueError):
        runtime.reset(options={"scope": scope, "vessel_ids": ["treatment_reactor"]})
    assert runtime._state.to_dict() == before


def test_actual_numerical_failure_rolls_back_original_three_tanks(monkeypatch):
    import chemworld.runtime.reactive_water_treatment as adapter

    runtime = setup(shared=True)
    commit(runtime, {"operation": "set_flow_rate", "flow_rate_mL_min": 10.0})
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected 49B integration failure")

    monkeypatch.setattr(adapter, "advance_reactive_water_treatment", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 5.0})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.species_amounts == before.species_amounts
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.inactive_vessels == before.inactive_vessels
    assert runtime._state.ledger.time_s == before.ledger.time_s
