"""Declared native strategies retain their own equations on one physical clock."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.examples.closed_shared_labs import phase_closed_lab, reactive_closed_lab
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states


def session(spec, noise):
    runtime = ChemWorldRuntime(
        composition=spec, observation_noise_mode=noise, observation_seed_override=12,
    )
    runtime.reset(seed=7)
    return runtime


def execute(runtime, operation, **fields):
    result = runtime.execute({"operation": operation, **fields})
    assert result.info["transaction_status"] == "committed", {
        key: result.info.get(key) for key in (
            "transaction_status", "rollback_reason", "outcome_diagnostics",
        )
    }
    return result


def assert_local_physics(actual, expected):
    # These are the same local strategy/interval, not an independent science
    # reference. Check every owned physical stock and apparatus without tolerance.
    for field in (
        "species_amounts", "temperature_K", "pressure_Pa", "volume_L",
        "phases", "state_slots", "equipment",
    ):
        assert getattr(actual, field) == getattr(expected, field), field
    assert actual.process.stochastic_continuation == expected.process.stochastic_continuation


@pytest.mark.parametrize("factory", (reactive_closed_lab, phase_closed_lab))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_heterogeneous_wait_matches_each_local_strategy_at_the_common_endpoint(factory, noise):
    runtime = session(factory(), noise)
    try:
        advancer = runtime.runtime.domain_services.reaction_thermal.advancer
        for duration in (1, 2, 5):
            before = local_states(runtime._state)
            expected = {
                key: advancer.batch(local, {"operation": "wait", "duration_s": duration},
                                    heat=False)
                for key, local in before.items()
            }
            result = execute(runtime, "wait", duration_s=duration)
            assert result.info["operation_outcome"] == "success"
            assert runtime._state.vessel_elapsed_s == dict.fromkeys(
                before, runtime._state.ledger.time_s,
            )
            for key, local in local_states(runtime._state).items():
                assert expected[key].outcome == "success"
                assert expected[key].actual_duration_s == duration
                assert_local_physics(local, expected[key].state)
        if factory is reactive_closed_lab:
            local = local_states(runtime._state)
            assert local["reactor"].species_amounts["B"] > 0
            assert local["cell"].state_slots.slots[0].values["discharged_charge_C"] > 0
            assert local["adsorber"].state_slots != factory().physics.initial_states[
                "adsorber"].state_slots
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_thermal_battery_accepts_the_one_actual_heater(noise):
    runtime = session(reactive_closed_lab(), noise)
    try:
        initial = runtime._state.inactive_vessels["cell"].temperature_K
        execute(runtime, "select_vessel", vessel="cell")
        execute(runtime, "queue_lab_heat", target_temperature_K=310, duration_s=2)
        execute(runtime, "wait", duration_s=2)
        assert runtime._state.temperature_K > initial
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(runtime._state.vessel_elapsed_s, 2)
        assert [key for key, local in local_states(runtime._state).items()
                if "adsorber:heater" in local.equipment.equipment] == ["cell"]
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_battery_cutoff_aligns_reaction_adsorption_and_ssa_to_actual_time(noise):
    from chemworld.physchem.battery_models import (
        BatterySPMSpec,
        BatteryThermalSpec,
        battery_readout,
    )

    battery = BatterySPMSpec(thermal=BatteryThermalSpec())
    initial_voltage = battery_readout(battery, battery.initial_state(), .02).voltage_V
    battery = replace(battery, minimum_voltage_V=initial_voltage - .001)
    runtime = session(reactive_closed_lab(battery_spec=battery), noise)
    try:
        before = local_states(runtime._state)
        result = execute(runtime, "wait", duration_s=100)
        assert result.info["operation_outcome"] == "stopped"
        elapsed = runtime._state.ledger.time_s
        assert 0 < elapsed < 100
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, elapsed)
        advancer = runtime.runtime.domain_services.reaction_thermal.advancer
        for key, local in local_states(runtime._state).items():
            expected = advancer.batch(before[key], {"duration_s": elapsed}, heat=False)
            assert expected.outcome in {"success", "stopped"}
            assert_local_physics(local, expected.state)
        assert runtime._state.lab_clock.stopped
    finally:
        runtime.close()


def test_one_heater_changes_both_crystal_inventory_and_solid_equilibrium():
    runtime = session(phase_closed_lab(), "sequential")
    try:
        initial = local_states(runtime._state)
        execute(runtime, "queue_lab_heat", target_temperature_K=350, duration_s=5)
        execute(runtime, "wait", duration_s=5)
        crystallizer = local_states(runtime._state)["crystallizer"]
        assert crystallizer.phases != initial["crystallizer"].phases
        assert all(n >= 0 for phase in crystallizer.phases.phases.values()
                   for n in phase.species_amounts_mol.values())
        execute(runtime, "select_vessel", vessel="specimen")
        execute(runtime, "queue_lab_heat", target_temperature_K=450, duration_s=30)
        execute(runtime, "wait", duration_s=30)
        assert runtime._state.phases != initial["specimen"].phases
        assert runtime._state.species_amounts == initial["specimen"].species_amounts
        assert runtime._state.vessel_elapsed_s == {"crystallizer": 35, "specimen": 35}
    finally:
        runtime.close()


def test_isothermal_battery_has_shared_time_but_no_heat_capability():
    from chemworld.physchem.battery_models import BatterySPMSpec

    runtime = session(reactive_closed_lab(battery_spec=BatterySPMSpec()), "sequential")
    try:
        execute(runtime, "select_vessel", vessel="cell")
        before = local_states(runtime._state)
        result = runtime.execute({
            "operation": "queue_lab_heat", "target_temperature_K": 310, "duration_s": 2,
        })
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.ledger.time_s == 0
        for key, local in local_states(runtime._state).items():
            assert_local_physics(local, before[key])
        execute(runtime, "wait", duration_s=1)
    finally:
        runtime.close()


def test_locally_installed_strategy_requires_explicit_shared_clock_opt_in():
    from chemworld.world.lab_world import shared_lab_world
    from test_declared_advance_strategies import with_local_thermal

    spec = with_local_thermal(shared_lab_world())
    with pytest.raises(ValueError, match="shared-clock participation"):
        session(spec, "sequential")
    modules = replace(spec.modules, modules=tuple(
        replace(module, advance_strategies=tuple(
            replace(strategy, shared_clock=True) if strategy.strategy_id == "local-thermal"
            else strategy for strategy in module.advance_strategies
        )) for module in spec.modules.modules
    ))
    runtime = session(replace(spec, modules=modules), "sequential")
    try:
        execute(runtime, "wait", duration_s=1)
    finally:
        runtime.close()


def test_heat_queue_checks_the_actual_device_range_before_creating_a_job():
    from chemworld.foundation.state_ledgers import EquipmentLedger

    spec = phase_closed_lab()
    states = dict(spec.physics.initial_states)
    local = states["specimen"]
    records = dict(local.equipment.equipment)
    device = records["specimen:heater"]
    records[device.equipment_id] = replace(device, settings={
        **device.settings, "thermal_device": {
            **device.settings["thermal_device"], "maximum_setpoint_K": 320,
        },
    })
    states["specimen"] = local.replace(equipment=EquipmentLedger(records))
    runtime = session(replace(spec, physics=replace(spec.physics, initial_states=states)),
                      "sequential")
    try:
        execute(runtime, "select_vessel", vessel="specimen")
        lab = runtime._state.lab_clock
        result = runtime.execute({
            "operation": "queue_lab_heat", "target_temperature_K": 330, "duration_s": 5,
        })
        assert result.info["transaction_status"] == "rolled_back"
        assert runtime._state.lab_clock == lab
        assert runtime._state.ledger.time_s == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("factory", (reactive_closed_lab, phase_closed_lab))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_heterogeneous_owned_state_restores_in_a_new_process(factory, noise, tmp_path):
    runtime = session(factory(), noise)
    restored = None
    try:
        execute(runtime, "wait", duration_s=1)
        source, output = tmp_path / "snapshot.json", tmp_path / "continued.json"
        runtime.snapshot().save(source)
        restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(
            runtime.snapshot().to_dict(), sort_keys=True,
        )))
        for duration in (2, 3):
            execute(runtime, "wait", duration_s=duration)
            execute(restored, "wait", duration_s=duration)
        assert runtime._state.to_dict() == restored._state.to_dict()
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
            "results=[r.execute({'operation':'wait','duration_s':t}) for t in (2,3)]; "
            "assert all(x.info['transaction_status']=='committed' for x in results); "
            f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
        assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
    finally:
        runtime.close()
        if restored:
            restored.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_heterogeneous_failure_discards_sampling_and_every_owned_candidate(noise, monkeypatch):
    from chemworld.runtime.kernel_contracts import ModelProviderResult
    from test_finite_lab_clock import assert_only_rejection_penalty

    runtime = session(reactive_closed_lab(), noise)
    try:
        execute(runtime, "select_vessel", vessel="ssa")
        execute(runtime, "wait", duration_s=1)
        execute(runtime, "queue_lab_measure")
        execute(runtime, "queue_lab_measure")
        before = runtime._state.to_dict()
        solver = runtime.runtime.domain_services.reaction_thermal.advancer.strategies[
            "single-particle-battery"]

        def failure(*args, **kwargs):
            return ModelProviderResult(
                {}, diagnostics={"injected": True}, state=None, success=False,
                outcome="numerical_failure", failure_reason="injected_battery_failure",
            )

        monkeypatch.setattr(solver, "advance", failure)
        result = runtime.execute({"operation": "wait", "duration_s": 5})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert_only_rejection_penalty(before, runtime._state.to_dict())
    finally:
        runtime.close()
