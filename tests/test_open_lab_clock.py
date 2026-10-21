"""A whole finite apparatus is one shared advance, not one advance per tank."""

import json
import os
import subprocess
import sys

import pytest

from chemworld.examples.open_shared_labs import open_shared_lab
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.world.composition import WorldCompositionSpec


def session(kind, noise="sequential", **parameters):
    runtime = ChemWorldRuntime(
        composition=open_shared_lab(kind, **parameters), observation_noise_mode=noise,
        observation_seed_override=12,
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


@pytest.mark.parametrize("kind", ("transport", "membrane"))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_open_system_and_closed_bottle_advance_together_once(kind, noise):
    runtime = session(kind, noise)
    try:
        initial = local_states(runtime._state)
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        execute(runtime, "select_vessel", vessel="bottle")
        execute(runtime, "queue_lab_measure")
        result = execute(runtime, "wait", duration_s=5)
        assert result.info["outcome_diagnostics"]["systems_advanced"] == 2
        assert result.info["outcome_diagnostics"]["vessels_advanced"] == len(initial)
        systems = result.info["outcome_diagnostics"]["shared_systems"]
        assert systems["bottle"]["vessels"] == ["bottle"]
        assert set(systems["reactor"]["vessels"]) == set(initial) - {"bottle"}
        assert len(systems["reactor"]["connections"]) == (1 if kind == "transport" else 3)
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(initial, 5)
        assert runtime._state.temperature_K < initial["bottle"].temperature_K
        assert runtime._state.ledger.analysis_time_s == 5
        assert runtime._state.ledger.sample_consumed_L == pytest.approx(.0002)
        assert runtime._state.ledger.flow_pump_work_J > 0
        current = local_states(runtime._state)
        if kind == "transport":
            assert current["feed"].volume_L < initial["feed"].volume_L
        else:
            assert current["reactor"].volume_L < initial["reactor"].volume_L
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("transport", "membrane"))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_shared_open_system_matches_the_existing_local_session(kind, noise):
    from chemworld.physchem.membranes import MembraneFoulingSpec, MembraneSpec
    from chemworld.world.membrane_world import membrane_world_spec
    from chemworld.world.transport_world import transport_world_spec
    from test_heterogeneous_lab_clock import assert_local_physics

    world = transport_world_spec() if kind == "transport" else membrane_world_spec(
        MembraneSpec(fouling=MembraneFoulingSpec()),
    )
    shared = session(kind, noise)
    local = ChemWorldRuntime(composition=world, observation_noise_mode=noise,
                             observation_seed_override=12)
    try:
        local.reset(seed=7)
        for runtime in (shared, local):
            execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        for duration in (5, 7, 11):
            execute(shared, "wait", duration_s=duration)
            execute(local, "wait", duration_s=duration)
            current = local_states(shared._state)
            for key, expected in local_states(local._state).items():
                assert_local_physics(current[key], expected)
            assert shared._state.ledger.flow_pump_work_J == local._state.ledger.flow_pump_work_J
            assert shared._state.ledger.cost == local._state.ledger.cost
            bottle = current["bottle"].thermal.vessels["bottle"]
            assert shared._state.ledger.energy_jacket_J == local._state.ledger.energy_jacket_J
            assert shared._state.ledger.heat_loss_J == pytest.approx(
                local._state.ledger.heat_loss_J + bottle.heat_loss_J, abs=1e-6,
            )
    finally:
        shared.close()
        local.close()


@pytest.mark.parametrize("kind", ("transport", "membrane"))
def test_finite_open_boundary_aligns_the_closed_bottle(kind):
    from chemworld.physchem.membranes import MembraneFoulingSpec, MembraneSpec

    parameters = {"feed_volume_L": .001} if kind == "transport" else {
        "model": MembraneSpec(collector_capacity_L=.0001, fouling=MembraneFoulingSpec()),
    }
    runtime = session(kind, **parameters)
    try:
        before = local_states(runtime._state)["bottle"]
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        result = execute(runtime, "wait", duration_s=1000)
        assert result.info["operation_outcome"] == "stopped"
        endpoint = runtime._state.ledger.time_s
        assert 0 < endpoint < 1000
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(runtime._state.vessel_elapsed_s,
                                                               endpoint)
        expected = runtime.runtime.domain_services.reaction_thermal.advancer.batch(
            before, {"duration_s": endpoint}, heat=False,
        )
        assert local_states(runtime._state)["bottle"].temperature_K == expected.state.temperature_K
        assert runtime._state.lab_clock.stopped
        events = result.info["outcome_diagnostics"]["stopping_events"]
        assert events["reactor"]["stopping_event"] == (
            "feed_pickup_exhausted" if kind == "transport" else "collector_full"
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_finite_membrane_rinse_debits_wash_and_moves_bound_foulant_to_waste(noise):
    from chemworld.physchem.membranes import MembraneSpec
    from chemworld.runtime.membranes import membrane_record

    runtime = session("membrane", noise, wash_volume_L=.001)
    try:
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        execute(runtime, "wait", duration_s=5)
        before = local_states(runtime._state)
        model = MembraneSpec.from_dict(
            membrane_record(before["reactor"]).settings["physical_model"],
        )
        wash, waste = model.fouling.wash_vessel_id, model.fouling.waste_vessel_id
        bound = next(slot.values["bound_mol"] for slot in before["reactor"].state_slots.slots
                     if slot.spec.state_type == "membrane_bound_foulant_v1")
        assert bound > 0
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20, flow_path="rinse")
        result = execute(runtime, "wait", duration_s=1000)
        assert result.info["operation_outcome"] == "stopped"
        assert result.info["outcome_diagnostics"]["stopping_events"]["reactor"][
            "stopping_event"] == "wash_pickup_exhausted"
        current = local_states(runtime._state)
        assert current[wash].volume_L < before[wash].volume_L
        assert current[waste].volume_L > before[waste].volume_L
        assert current[waste].species_amounts["foulant"] > 0
        remaining = next(slot.values["bound_mol"]
                         for slot in current["reactor"].state_slots.slots
                         if slot.spec.state_type == "membrane_bound_foulant_v1")
        assert remaining < bound
        assert len(set(runtime._state.vessel_elapsed_s.values())) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("transport", "membrane"))
def test_zero_flow_still_advances_the_independent_bottle_without_creating_work(kind):
    runtime = session(kind)
    try:
        execute(runtime, "set_flow_rate", flow_rate_mL_min=0)
        before = local_states(runtime._state)
        execute(runtime, "wait", duration_s=5)
        current = local_states(runtime._state)
        assert current["bottle"].temperature_K < before["bottle"].temperature_K
        for key in set(current) - {"bottle"}:
            assert current[key].species_amounts == before[key].species_amounts
        assert runtime._state.ledger.flow_pump_work_J == 0
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(current, 5)
    finally:
        runtime.close()


def test_shared_world_cannot_enable_a_second_local_run_flow_clock():
    spec = open_shared_lab("transport")
    payload = spec.to_dict()
    payload["task"]["operations"].append("run_flow")
    with pytest.raises(ValueError, match="run_flow"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


@pytest.mark.parametrize("maximum_head,maximum_power", ((1000, 10), (1e6, 1e-4)))
def test_actual_pump_head_and_power_limits_are_settled_once(maximum_head, maximum_power):
    from chemworld.physchem.transport import PumpSpec
    from chemworld.physchem.transport_devices import FiniteTransportSpec
    from chemworld.world.transport_world import synthetic_transport_properties

    spec = FiniteTransportSpec(
        synthetic_transport_properties(), pump=PumpSpec(maximum_head, maximum_power, .7),
    )
    runtime = session("transport", model=spec)
    try:
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        result = execute(runtime, "wait", duration_s=10)
        diagnostic = result.info["outcome_diagnostics"]["local_outcomes"]["reactor"]["transport"]
        assert diagnostic["actual_flow_m3_s"] < 20 / 6e7
        assert diagnostic["pressure_rise_Pa"] <= maximum_head * (1 + 1e-8)
        assert diagnostic["shaft_power_W"] <= maximum_power * (1 + 1e-8)
        assert runtime._state.ledger.flow_pump_work_J == diagnostic["shaft_work_J"]
        assert runtime._state.ledger.cost == diagnostic["shaft_work_J"] / 250000
        assert runtime._state.vessel_elapsed_s["bottle"] == 10
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("transport", "membrane"))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_open_system_failure_discards_all_tanks_sampling_and_rng(kind, noise, monkeypatch):
    from chemworld.runtime.kernel_contracts import ModelProviderResult
    from test_finite_lab_clock import assert_only_rejection_penalty

    runtime = session(kind, noise)
    try:
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        execute(runtime, "wait", duration_s=1)
        execute(runtime, "select_vessel", vessel="bottle")
        execute(runtime, "queue_lab_measure")
        before = runtime._state.to_dict()
        identity = "finite-pump-transport" if kind == "transport" else "finite-dilute-ro"
        solver = runtime.runtime.domain_services.reaction_thermal.advancer.strategies[identity]

        def failure(*args, **kwargs):
            return ModelProviderResult(
                {}, diagnostics={"injected": True}, state=None, success=False,
                outcome="numerical_failure", failure_reason="injected_connected_failure",
            )

        monkeypatch.setattr(solver, "advance", failure)
        result = runtime.execute({"operation": "wait", "duration_s": 5})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert_only_rejection_penalty(before, runtime._state.to_dict())
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("transport", "membrane"))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_connected_system_snapshot_keeps_analysis_and_flow_in_a_new_process(kind, noise, tmp_path):
    runtime = session(kind, noise)
    try:
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        execute(runtime, "select_vessel", vessel="bottle")
        execute(runtime, "queue_lab_measure")
        execute(runtime, "wait", duration_s=5)
        assert len(runtime._state.lab_clock.running) == 1
        source, output = tmp_path / "session.json", tmp_path / "continued.json"
        runtime.snapshot().save(source)
        for duration in (7, 60):
            execute(runtime, "wait", duration_s=duration)
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
            "results=[r.execute({'operation':'wait','duration_s':t}) for t in (7,60)]; "
            "assert all(x.info['transaction_status']=='committed' for x in results); "
            f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
        assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("transport", "membrane"))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_connected_assembly_is_independent_of_enumeration_and_selection(kind, noise):
    from test_finite_lab_clock import physical_records

    normal = open_shared_lab(kind)
    payload = normal.to_dict()
    payload["vessels"].reverse()
    payload["components"].reverse()
    payload["connections"].reverse()
    payload["physics"]["initial_states"] = dict(reversed(
        payload["physics"]["initial_states"].items(),
    ))
    runtimes = [ChemWorldRuntime(composition=spec, observation_noise_mode=noise,
                                observation_seed_override=12)
                for spec in (normal, WorldCompositionSpec.from_dict(payload))]
    try:
        for index, runtime in enumerate(runtimes):
            runtime.reset(seed=7)
            execute(runtime, "select_vessel", vessel="reactor")
            execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
            if index:
                auxiliary = "feed" if kind == "transport" else "permeate"
                execute(runtime, "select_vessel", vessel=auxiliary)
            for duration in (5, 7):
                execute(runtime, "wait", duration_s=duration)
            execute(runtime, "select_vessel", vessel="reactor")
        assert physical_records(runtimes[0]) == physical_records(runtimes[1])
        assert runtimes[0]._state.ledger == runtimes[1]._state.ledger
        assert runtimes[0]._state.solvent_accounting == runtimes[1]._state.solvent_accounting
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.parametrize("mutation", ("extra_path", "duplicate_path", "overlap"))
def test_unsupported_connected_topology_is_rejected_before_execution(mutation):
    spec = open_shared_lab("transport")
    payload = spec.to_dict()
    if mutation == "overlap":
        payload["physics"]["models"]["thermal"]["bindings"]["feed"] = payload[
            "physics"]["models"]["thermal"]["bindings"]["bottle"]
    else:
        edge = {**payload["connections"][0], "id": "unsupported-extra"}
        if mutation == "extra_path":
            edge = {**edge, "source": edge["target"], "target": edge["source"]}
        payload["connections"].append(edge)
    with pytest.raises(ValueError, match=r"overlap|unambiguous|unsupported|port"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
