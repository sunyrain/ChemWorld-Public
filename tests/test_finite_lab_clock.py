"""Finite declared systems exercise the actual session and transaction boundary."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.native_domains import mixed_clock_world
from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.runtime.stochastic import StochasticRuntime, install_stochastic_state
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.lab_world import shared_lab_world


def finite_world(count, *, mixed=False, reverse=False):
    """Distinct authored charges; no runtime inventory or device duplication."""
    payload = (mixed_clock_world() if mixed else shared_lab_world()).to_dict()
    physics = payload["physics"]
    states = physics["initial_states"]
    if count == 1:
        keep = next(iter(states))
        payload["vessels"] = [v for v in payload["vessels"] if v["id"] == keep]
        payload["components"] = [c for c in payload["components"] if c["vessel"] == keep]
        payload["task"]["operations"].remove("select_vessel")
        physics["initial_states"] = states = {keep: states[keep]}
        if "thermal" in physics["models"]:
            bindings = physics["models"]["thermal"]["bindings"]
            physics["models"]["thermal"]["bindings"] = {
                key: value for key, value in bindings.items() if key == keep
            }
    if count == 4:
        for index, name in enumerate(("vial-c", "vial-d"), 1):
            discrete = mixed and name == "vial-d"
            source = WorldState.from_dict(states["reactor"])
            local = WorldState(
                {key: amount * (index + 1) for key, amount in source.species_amounts.items()},
                source.volume_L if discrete else .01 * (index + 2),
                source.temperature_K if discrete else 315 + 7 * index,
                101325, "liquid", name, units=source.units, equipment=EquipmentLedger({}),
            )
            if discrete:
                local = install_stochastic_state(local, seed=37)
            else:
                bindings = physics["models"]["thermal"]["bindings"]
                bindings[name] = deepcopy(next(iter(bindings.values())))
            states[name] = local.to_dict()
            payload["vessels"].append({"id": name, "capacity_L": 0.1})
            payload["components"] += [
                {
                    "id": name + ":" + kind, "vessel": name, "kind": kind,
                    "parameters": {"duration_range_s": [1e-6, 14400]}
                    if kind == "lab_schedule" else {
                        "instruments": ["molecule_capture" if mixed else "programmed_hplc"],
                    } if kind == "observation" else {},
                }
                for kind in (("stochastic", "observation", "lab_schedule") if discrete
                             else ("thermal", "lab_schedule") if mixed
                             else ("thermal", "observation", "lab_schedule"))
            ]
    physics["lab_clock"]["vessel_capacity"] = count
    if reverse:
        payload["vessels"].reverse()
        payload["components"].reverse()
        physics["initial_states"] = dict(reversed(states.items()))
        for model in physics["models"].values():
            model["bindings"] = dict(reversed(model["bindings"].items()))
    return WorldCompositionSpec.from_dict(payload)


def session(spec, noise="sequential"):
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


def physical_records(runtime):
    records = {}
    for key, local in local_states(runtime._state).items():
        record = local.to_dict()
        # The declaration order is descriptive metadata, not physical inventory.
        record.pop("metadata")
        records[key] = record
    return records


def assert_only_rejection_penalty(before, after):
    """Rejection history remains charged, while the proposed physical slice vanishes."""
    before, after = deepcopy(before), deepcopy(after)
    assert after["ledger"]["cost"] == pytest.approx(before["ledger"]["cost"] + .03)
    assert after["ledger"]["risk"] == pytest.approx(min(1, before["ledger"]["risk"] + .08))
    for key in ("ledger", "process"):
        for field in ("cost", "risk"):
            before[key].pop(field)
            after[key].pop(field)
    assert before == after


@pytest.mark.parametrize("count", (1, 2, 4))
@pytest.mark.parametrize("mixed", (False, True))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_all_declared_systems_evolve_independently_of_enumeration(count, mixed, noise):
    runtimes = [session(finite_world(count, mixed=mixed, reverse=reverse), noise)
                for reverse in (False, True)]
    try:
        for index, runtime in enumerate(runtimes):
            if count > 1:
                execute(runtime, "select_vessel", vessel="reactor")
            initial = physical_records(runtime)
            execute(runtime, "queue_lab_measure")
            execute(runtime, "queue_lab_measure")
            execute(runtime, "wait", duration_s=2)
            assert runtime._state.vessel_elapsed_s == dict.fromkeys(initial, 1 if mixed else 2)
            for key, local in local_states(runtime._state).items():
                if local.process.stochastic_continuation:
                    assert local.process.stochastic_continuation["time_s"] == 1
                elif initial[key]["temperature_K"] > 300:
                    assert local.temperature_K < initial[key]["temperature_K"]
            if count > 1 and index:
                execute(runtime, "select_vessel", vessel=sorted(initial)[-1])
            execute(runtime, "wait", duration_s=5)
            if count > 1:
                execute(runtime, "select_vessel", vessel="reactor")
        left, right = runtimes
        assert physical_records(left) == physical_records(right)
        assert left._state.ledger == right._state.ledger
        assert left._state.lab_clock == right._state.lab_clock
        assert left._state.solvent_accounting == right._state.solvent_accounting
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.parametrize("value", (True, 0, -1, 1.0, 65))
def test_system_capacity_is_a_finite_explicit_integer(value):
    with pytest.raises(ValueError, match="vessel capacity"):
        LabClockSpec(vessel_capacity=value)


def test_capacity_rejects_an_overfull_world_before_any_execution():
    spec = finite_world(4)
    limited = replace(spec, physics=replace(
        spec.physics, lab_clock=replace(spec.physics.lab_clock, vessel_capacity=2),
    ))
    with pytest.raises(ValueError, match="vessel capacity"):
        session(limited)
    assert WorldCompositionSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_four_system_failure_rolls_back_acquisition_rng_and_whole_slice(noise, monkeypatch):
    runtime = session(finite_world(4, mixed=True), noise)
    try:
        execute(runtime, "wait", duration_s=2)
        execute(runtime, "queue_lab_measure")
        execute(runtime, "queue_lab_measure")
        before = runtime._state.to_dict()
        solver = runtime.runtime.domain_services.reaction_thermal.advancer
        actual = solver.batch
        attempted = []

        def fail_last(local, action, **kwargs):
            from chemworld.runtime.kernel_contracts import ModelProviderResult

            attempted.append(local.vessel_id)
            if local.vessel_id == "vial-d":
                return ModelProviderResult(
                    {}, diagnostics={}, state=None, success=False, outcome="numerical_failure",
                    failure_reason="injected_fourth_system_failure",
                )
            return actual(local, action, **kwargs)

        monkeypatch.setattr(solver, "batch", fail_last)
        result = runtime.execute({"operation": "wait", "duration_s": 5})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert set(attempted) == set(before["vessel_elapsed_s"])
        assert_only_rejection_penalty(before, runtime._state.to_dict())
        monkeypatch.setattr(solver, "batch", actual)
        execute(runtime, "wait", duration_s=5)
        assert runtime._state.ledger.time_s == 3
        assert len(runtime._state.lab_clock.completed) == 1
    finally:
        runtime.close()


def test_four_system_earliest_stop_aligns_all_clocks_and_keeps_committed_history():
    spec = finite_world(4)
    thermal = dict(spec.physics.models_of("thermal"))
    thermal["reactor"] = replace(thermal["reactor"], stop_temperature_K=299)
    spec = replace(spec, physics=replace(spec.physics, thermal_models=thermal))
    runtime = session(spec)
    try:
        execute(runtime, "wait", duration_s=1)
        committed = runtime._state.to_dict()
        execute(runtime, "queue_lab_heat", target_temperature_K=350, duration_s=30)
        execute(runtime, "queue_lab_measure")
        result = execute(runtime, "wait", duration_s=120)
        assert result.info["operation_outcome"] == "stopped"
        endpoint = runtime._state.ledger.time_s
        assert 1 < endpoint < 31
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(
            committed["vessel_elapsed_s"], endpoint,
        )
        assert runtime._state.temperature_K == pytest.approx(299, abs=1e-6)
        assert runtime._state.ledger.analysis_time_s == pytest.approx(endpoint - 1)
        assert runtime._state.lab_clock.stopped
        stopped = runtime._state.to_dict()
        assert runtime.execute({"operation": "wait", "duration_s": 1}).info[
            "operation_outcome"] == "rejected"
        assert_only_rejection_penalty(stopped, runtime._state.to_dict())
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_thermal_stop_rewinds_ssa_candidates_to_the_actual_common_endpoint(noise):
    spec = finite_world(4, mixed=True)
    states = dict(spec.physics.initial_states)
    local = states["continuous"]
    states["continuous"] = local.replace(equipment=EquipmentLedger({
        "heater": EquipmentRecord(
            "heater", "thermal_device", "continuous",
            settings={"thermal_device": ThermalDeviceSpec(enabled=False).to_dict()},
        ),
    }))
    thermal = dict(spec.physics.models_of("thermal"))
    thermal["continuous"] = replace(thermal["continuous"], stop_temperature_K=351)
    spec = replace(
        spec,
        physics=replace(
            spec.physics, initial_states=states, thermal_models=thermal,
            lab_clock=replace(spec.physics.lab_clock, heater_id="heater"),
        ),
        task=replace(spec.task, operations=(*spec.task.operations, "queue_lab_heat")),
    )
    runtime = session(spec, noise)
    try:
        execute(runtime, "wait", duration_s=2)
        execute(runtime, "select_vessel", vessel="continuous")
        before = local_states(runtime._state)
        execute(runtime, "queue_lab_heat", target_temperature_K=370, duration_s=30)
        result = execute(runtime, "wait", duration_s=30)
        assert result.info["operation_outcome"] == "stopped"
        endpoint = runtime._state.ledger.time_s
        assert 2 < endpoint < 32
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, endpoint)
        for key, local in local_states(runtime._state).items():
            if local.process.stochastic_continuation:
                single = StochasticRuntime(spec).advance(
                    before[key], {"duration_s": endpoint - 2},
                )
                assert single.outcome == "success"
                assert local.species_amounts == single.state.species_amounts
                assert local.process.stochastic_continuation == \
                    single.state.process.stochastic_continuation
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_four_vessel_fifo_moves_the_unique_devices_by_persistent_sequence(noise):
    runtime = session(finite_world(4), noise)
    try:
        for key in sorted(runtime._state.vessel_elapsed_s):
            execute(runtime, "select_vessel", vessel=key)
            execute(runtime, "queue_lab_heat", target_temperature_K=350, duration_s=60)
            execute(runtime, "queue_lab_measure")
        for _ in range(4):
            execute(runtime, "wait", duration_s=100)
        jobs = runtime._state.lab_clock.completed
        assert [job["sequence"] for job in jobs] == list(range(8))
        assert [job["available_at"] for job in jobs] == [60, 60, 120, 120, 180, 180, 240, 240]
        owners = [key for key, local in local_states(runtime._state).items()
                  if "heater" in local.equipment.equipment]
        assert owners == ["vial-d"]
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(
            runtime._state.vessel_elapsed_s, 240,
        )
        assert runtime._state.ledger.analysis_time_s == 240
        assert runtime._state.ledger.cost == pytest.approx(.2)
        assert runtime._state.ledger.sample_consumed_L == pytest.approx(.0008)
        assert sum("instrument:programmed_hplc" in local.equipment.equipment
                   for local in local_states(runtime._state).values()) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_four_mixed_systems_resume_pending_queue_and_ssa_in_new_process(tmp_path, noise):
    runtime = session(finite_world(4, mixed=True, reverse=True), noise)
    try:
        execute(runtime, "select_vessel", vessel="reactor")
        execute(runtime, "queue_lab_measure")
        execute(runtime, "queue_lab_measure")
        execute(runtime, "wait", duration_s=0.25)
        lab = runtime._state.lab_clock
        assert len(lab.running) == len(lab.queued) == 1
        assert "observation" not in runtime.lab_status()["running"][0]
        for local in local_states(runtime._state).values():
            if local.process.stochastic_continuation:
                assert local.process.stochastic_continuation["pending"] is not None
        source, output = tmp_path / "session.json", tmp_path / "continued.json"
        runtime.snapshot().save(source)
        actions = [{"operation": "wait", "duration_s": t} for t in (.75, 1, 3)]
        for action in actions:
            execute(runtime, **action)
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
            f"results=[r.execute(a) for a in {actions!r}]; "
            "assert all(x.info['transaction_status']=='committed' for x in results); "
            f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict(),sort_keys=True))"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
        assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
    finally:
        runtime.close()
