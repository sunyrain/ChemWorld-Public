"""Public session paths, not a surrogate scheduler or fabricated observations."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.lab_world import shared_lab_world


def make(spec=None):
    runtime = ChemWorldRuntime(composition=spec or shared_lab_world())
    runtime.reset(seed=17)
    return runtime


def execute(runtime, operation, **fields):
    result = runtime.execute({"operation": operation, **fields})
    assert result.info["transaction_status"] == "committed", {
        k: result.info.get(k)
        for k in (
            "transaction_status",
            "rollback_reason",
            "outcome_diagnostics",
            "constitution_checks",
        )
    }
    return result


def select(runtime, vessel):
    execute(runtime, "select_vessel", vessel=vessel)


def locals_(runtime):
    return {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}


def enqueue_both(runtime):
    execute(runtime, "queue_lab_heat", duration_s=30, target_temperature_K=350)
    execute(runtime, "queue_lab_measure")
    select(runtime, "sample")
    execute(runtime, "queue_lab_heat", duration_s=30, target_temperature_K=360)
    execute(runtime, "queue_lab_measure")


def test_fifo_same_timestamp_shared_aging_and_single_settlement():
    r = make()
    enqueue_both(r)
    execute(r, "wait", duration_s=120)
    assert r._state.ledger.time_s == 30
    assert r._state.vessel_elapsed_s == {"reactor": 30, "sample": 30}
    assert locals_(r)["reactor"].temperature_K > 300
    assert 298.15 < locals_(r)["sample"].temperature_K < 299
    assert [j["sequence"] for j in r._state.lab_clock.completed] == [0]
    assert "heater" in locals_(r)["reactor"].equipment.equipment
    result = execute(r, "wait", duration_s=120)
    assert r._state.ledger.time_s == 60
    assert [j["sequence"] for j in r._state.lab_clock.completed] == [0, 1, 2]
    assert "heater" in locals_(r)["sample"].equipment.equipment
    assert len(result.info["lab_clock"]["completed"]) == 3
    execute(r, "wait", duration_s=120)
    assert r._state.ledger.time_s == 120
    assert r._state.ledger.analysis_time_s == 120
    assert r._state.ledger.cost == pytest.approx(0.1)
    assert r._state.ledger.sample_consumed_L == pytest.approx(0.0004)
    assert (
        sum("instrument:programmed_hplc" in s.equipment.equipment for s in locals_(r).values()) == 1
    )
    assert r._state.ledger.energy_jacket_J == pytest.approx(300)


def test_late_result_is_sampled_at_service_start_not_at_queue_or_release():
    r = make()
    enqueue_both(r)
    result = execute(r, "wait", duration_s=30)
    assert "observation" not in json.dumps(result.info["lab_clock"])
    assert not r._state.lab_clock.completed[0].get("observation")
    result = execute(r, "wait", duration_s=30)
    first = result.info["lab_clock"]["completed"][1]["observation"]
    assert first["sampled_at"] == {"clock_id": "lab", "time_s": 0}
    assert first["available_at"] == {"clock_id": "lab", "time_s": 60}
    assert not any(job["sequence"] == 3 for job in result.info["lab_clock"]["running"])
    execute(r, "wait", duration_s=10)
    running = r._state.lab_clock.running[0]
    assert running["sequence"] == 3 and running["started_at"] == 60
    assert running["observation"]["sampled_at"]["time_s"] == 60
    assert running["available_at"] == 120
    assert r._state.ledger.analysis_time_s == 70
    assert "observation" not in r._state.to_dict(include_hidden=False)["lab_clock"]["running"][0]


def test_snapshot_while_busy_preserves_queues_connections_noise_and_carryover():
    r = make()
    enqueue_both(r)
    execute(r, "wait", duration_s=5)
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(r.snapshot().to_dict())))
    for seconds in (120, 120, 120, 10):
        first = execute(r, "wait", duration_s=seconds)
        second = execute(restored, "wait", duration_s=seconds)
        assert r._state.to_dict() == restored._state.to_dict()
        assert first.info["lab_clock"] == second.info["lab_clock"]
    assert (
        locals_(r)["sample"].equipment.equipment["instrument:programmed_hplc"].settings["use_count"]
        == 2
    )


def test_solver_failure_rolls_back_both_vessels_sampling_and_rng(monkeypatch):
    r = make()
    execute(r, "wait", duration_s=5)
    enqueue_both(r)
    before = r._state
    solver = r.runtime.domain_services.reaction_thermal.advancer
    actual = solver.batch

    def failure(state, action, **kwargs):
        if state.vessel_id == "sample":
            from chemworld.runtime.kernel_contracts import ModelProviderResult

            return ModelProviderResult(
                {},
                state=None,
                diagnostics={"injected": True},
                success=False,
                outcome="numerical_failure",
                failure_reason="injected",
            )
        return actual(state, action, **kwargs)

    monkeypatch.setattr(solver, "batch", failure)
    result = r.execute({"operation": "wait", "duration_s": 120})
    assert result.info["operation_outcome"] == "numerical_failure", result.info[
        "outcome_diagnostics"
    ]
    assert r._state.lab_clock == before.lab_clock
    assert r._state.inactive_vessels == before.inactive_vessels
    assert r._state.species_amounts == before.species_amounts
    assert r._state.equipment == before.equipment
    assert r._state.ledger.time_s == 5
    assert r._state.ledger.sample_consumed_L == before.ledger.sample_consumed_L
    assert r._state.ledger.analysis_time_s == 0


def test_earliest_physical_stop_realigns_all_systems():
    spec = shared_lab_world()
    models = {
        **spec.physics.models_of('thermal'),
        "reactor": replace(spec.physics.models_of('thermal')["reactor"], stop_temperature_K=299.0),
    }
    r = make(
        replace(
            spec, physics=replace(spec.physics, models={**spec.physics.models, "thermal": models})
        )
    )
    enqueue_both(r)
    result = execute(r, "wait", duration_s=120)
    assert result.info["operation_outcome"] == "stopped"
    assert 0 < r._state.ledger.time_s < 30
    assert len(set(r._state.vessel_elapsed_s.values())) == 1
    assert locals_(r)["reactor"].temperature_K == pytest.approx(299, abs=1e-6)
    assert r._state.ledger.analysis_time_s == pytest.approx(r._state.ledger.time_s)
    assert not r._state.lab_clock.completed
    assert r._state.lab_clock.stopped
    assert r.execute({"operation": "wait", "duration_s": 1}).info["operation_outcome"] == "rejected"


def test_single_vessel_matches_local_mode_without_contention():
    from chemworld.world.composition import WorldCompositionSpec

    payload = shared_lab_world().to_dict()
    payload["vessels"] = payload["vessels"][:1]
    payload["components"] = [c for c in payload["components"] if c["vessel"] == "reactor"]
    del payload["physics"]["initial_states"]["sample"]
    del payload["physics"]["models"]["thermal"]["bindings"]["sample"]
    payload["task"]["operations"].remove("select_vessel")
    shared = make(WorldCompositionSpec.from_dict(payload))
    del payload["physics"]["lab_clock"]
    payload["components"] = [c for c in payload["components"] if c["kind"] != "lab_schedule"]
    payload["task"]["operations"] = ["wait", "heat", "measure", "terminate"]
    local = make(WorldCompositionSpec.from_dict(payload))
    execute(shared, "queue_lab_heat", duration_s=10, target_temperature_K=350)
    execute(shared, "wait", duration_s=10)
    execute(local, "heat", duration_s=10, target_temperature_K=350)
    assert shared._state.temperature_K == local._state.temperature_K
    assert shared._state.species_amounts == local._state.species_amounts
    assert shared._state.ledger == local._state.ledger
    assert local._state.lab_clock is None


def test_bounds_capacity_and_partial_reset_do_not_corrupt_pending_jobs():
    spec = shared_lab_world()
    r = make(
        replace(
            spec,
            physics=replace(spec.physics, lab_clock=LabClockSpec("heater", "programmed_hplc", 1)),
        )
    )
    execute(r, "queue_lab_measure")
    before = r._state.lab_clock
    assert r.execute({"operation": "queue_lab_measure"}).info["operation_outcome"] == "rejected"
    assert r._state.lab_clock == before
    pending = r.snapshot().to_dict()
    with pytest.raises(
        ValueError, match="queued/running jobs 0:instrument:programmed_hplc:reactor",
    ):
        r._initialize_entities("material", {})
    assert r.snapshot().to_dict() == pending
    r.reset(seed=17)
    assert r._state.lab_clock.next_sequence == 0


def test_unsupported_component_and_direct_measurement_are_not_silent_clock_mixes():
    from chemworld.world.composition import WorldCompositionError

    spec = shared_lab_world()
    with pytest.raises(WorldCompositionError, match="measure"):
        make(replace(spec, task=replace(spec.task, operations=(*spec.task.operations, "measure"))))
    with pytest.raises(WorldCompositionError, match="shared"):
        make(
            replace(
                spec, physics=replace(spec.physics, models={**spec.physics.models, "thermal": {}})
            )
        )


def hybrid_spec():
    from chemworld.examples.native_domains import mixed_clock_world

    return mixed_clock_world()


def test_independent_ssa_and_continuous_systems_share_clock_not_a_hybrid_solver():
    r, split = make(hybrid_spec()), make(hybrid_spec())
    for seconds in (10,):
        execute(r, "wait", duration_s=seconds)
    for seconds in (1, 4, 5):
        execute(split, "wait", duration_s=seconds)
    assert r._state.process.stochastic_continuation == split._state.process.stochastic_continuation
    assert r._state.species_amounts == split._state.species_amounts
    assert r._state.inactive_vessels["continuous"].species_amounts["B"] > 0
    assert r._state.inactive_vessels["continuous"].temperature_K < 350
    assert r._state.vessel_elapsed_s == {"reactor": 10, "continuous": 10}
    execute(r, "queue_lab_measure")
    result = execute(r, "wait", duration_s=10)
    assert r._state.ledger.time_s == 11
    assert result.info["lab_clock"]["completed"][0]["observation"]["available_at"]["time_s"] == 11
    assert result.info["material_balance"]["exchange_count"] >= 1


def test_ssa_future_and_sampling_rng_resume_in_a_new_process(tmp_path):
    import os
    import subprocess
    import sys

    r = make(hybrid_spec())
    execute(r, "wait", duration_s=2)
    execute(r, "queue_lab_measure")
    source, output = tmp_path / "snapshot.json", tmp_path / "after.json"
    source.write_text(json.dumps(r.snapshot().to_dict()))
    action = {"operation": "wait", "duration_s": 5}
    result = r.execute(action)
    assert result.info["transaction_status"] == "committed"
    code = (
        "import json; from pathlib import Path; from chemworld.runtime import ChemWorldRuntime; "
        f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
        f"r.execute({action!r}); Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
    assert json.loads(output.read_text()) == json.loads(json.dumps(r._state.to_dict()))


def test_actual_gym_log_replays_the_same_equipment_queue(tmp_path):
    from chemworld.agent_interface import agent_view_bundle
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.eval.verify import verify_records

    env = ChemWorldEnv(composition=shared_lab_world())
    env.reset(seed=17)
    task_info = {**env.task_info(), **env.evaluator_provenance()}
    path = tmp_path / "lab.jsonl"
    actions = (
        {"operation": "queue_lab_measure"},
        {"operation": "queue_lab_heat", "duration_s": 30, "target_temperature_K": 350},
        {"operation": "select_vessel", "vessel": "sample"},
        {"operation": "queue_lab_measure"},
        *({"operation": "wait", "duration_s": 120} for _ in range(3)),
    )
    with TrajectoryLogger(path) as logger:
        for step, action in enumerate(actions, 1):
            observation, reward, terminated, truncated, info = env.step(action)
            assert info["transaction_status"] == "committed"
            logger.log(
                task_info=task_info,
                step=step,
                action=action,
                observation=observation_to_json(observation),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"agent_id": "shared-lab-test"},
                agent_view=agent_view_bundle(env, observation, info),
            )
    report = verify_records(load_jsonl(path))
    assert report.verified, report


def test_invalid_detector_still_consumes_real_sample_and_analysis_time(monkeypatch):
    r = make()
    schedule = r.runtime.domain_services.lab_schedule
    definitions = schedule.modules.instrument_specs()

    def invalid_readout(*args, **kwargs):
        raise ValueError("injected detector domain failure")

    modified = {
        **definitions,
        "programmed_hplc": replace(definitions["programmed_hplc"], native_observer=invalid_readout),
    }
    monkeypatch.setattr(type(schedule.modules), "instrument_specs", lambda self, **kwargs: modified)
    execute(r, "queue_lab_measure")
    before = r._state
    execute(r, "wait", duration_s=5)
    assert r._state.volume_L < before.volume_L
    assert r.lab_status()["completed"] == []
    assert "readout_error" not in r.lab_status()["running"][0]
    execute(r, "wait", duration_s=100)
    completed = r.lab_status()["completed"][0]
    assert completed["measurement_status"] == "invalid"
    assert "observation" not in completed
    assert r._state.ledger.cost == pytest.approx(0.05)
    assert r._state.ledger.sample_consumed_L == pytest.approx(0.0002)
    assert r._state.ledger.analysis_time_s == 60


def test_public_polling_does_not_reveal_pending_readings_or_change_rng():
    r = make()
    execute(r, "queue_lab_measure")
    execute(r, "wait", duration_s=5)
    before = r._state.to_dict()
    for _ in range(10):
        r.observe()
        status = r.lab_status()
        assert "detector_rng" not in status
        assert "observation" not in status["running"][0]
    assert r._state.to_dict() == before


def test_hybrid_boundary_still_rejects_one_missing_molecule():
    from chemworld.foundation.balances import TransitionBalance
    from chemworld.foundation.discrete_inventory import counts_to_mol, mol_to_counts

    r = make(hybrid_spec())
    before = r._state
    counts = mol_to_counts(before.species_amounts)
    counts["A"] -= 1
    after = before.replace(species_amounts=counts_to_mol(counts))
    report = r.constitution.check_transition(before, after, TransitionBalance())
    assert not report.passed


def test_selecting_another_vessel_does_not_resample_ssa_future():
    first, second = make(hybrid_spec()), make(hybrid_spec())
    execute(first, "wait", duration_s=3)
    execute(second, "wait", duration_s=3)
    select(second, "continuous")
    execute(first, "wait", duration_s=7)
    execute(second, "wait", duration_s=7)
    assert locals_(first)["reactor"].species_amounts == locals_(second)["reactor"].species_amounts
    assert (locals_(first)["reactor"].process.stochastic_continuation ==
            locals_(second)["reactor"].process.stochastic_continuation)
    assert first._state.ledger.time_s == second._state.ledger.time_s == 10


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_detector_modes_keep_process_sampling_and_polling_independent(noise):
    sessions = [ChemWorldRuntime(composition=hybrid_spec(), observation_noise_mode=noise,
                                observation_seed_override=12) for _ in range(2)]
    try:
        for r in sessions:
            r.reset(seed=7)
            execute(r, "queue_lab_measure")
            execute(r, "queue_lab_measure")
            execute(r, "wait", duration_s=0.25)
        untouched = sessions[1]._state.to_dict()
        for _ in range(10):
            sessions[1].observe()
            assert "detector_noise" not in sessions[1].lab_status()
            assert "observation" not in sessions[1].lab_status()["running"][0]
        assert sessions[1]._state.to_dict() == untouched
        for r in sessions:
            execute(r, "wait", duration_s=0.75)
            execute(r, "wait", duration_s=1)
        assert sessions[0]._state.to_dict() == sessions[1]._state.to_dict()
        jobs = sessions[0]._state.lab_clock.completed
        assert [job["sequence"] for job in jobs] == [0, 1]
        if noise == "keyed":
            assert all(job["noise_provenance"]["mode"] == "keyed" for job in jobs)
            assert len({job["noise_provenance"]["noise_key_sha256"] for job in jobs}) == 2
    finally:
        for r in sessions:
            r.close()


def test_keyed_queued_detection_does_not_change_ssa_or_sampling():
    sessions = [ChemWorldRuntime(composition=hybrid_spec(), observation_noise_mode=mode,
                                observation_seed_override=12)
                for mode in ("sequential", "keyed")]
    try:
        for r in sessions:
            r.reset(seed=7)
            execute(r, "wait", duration_s=2)
            execute(r, "queue_lab_measure")
            execute(r, "wait", duration_s=1)
        left, right = (r._state for r in sessions)
        assert left.species_amounts == right.species_amounts
        assert left.process.stochastic_continuation == right.process.stochastic_continuation
        assert left.ledger == right.ledger
        assert left.lab_clock.exchanges == right.lab_clock.exchanges
    finally:
        for r in sessions:
            r.close()


def test_subsecond_wait_requires_declared_lab_resolution_and_stays_positive():
    declared, standard = make(hybrid_spec()), make()
    try:
        execute(declared, "wait", duration_s=0.25)
        assert standard.execute({"operation": "wait", "duration_s": 0.25}).info[
            "transaction_status"] == "validation_failed"
        before = declared._state
        for duration in (0.0, -1.0, 1e-8):
            assert declared.execute({"operation": "wait", "duration_s": duration}).info[
                "transaction_status"] == "validation_failed"
            assert declared._state.species_amounts == before.species_amounts
            assert declared._state.ledger.time_s == before.ledger.time_s
    finally:
        declared.close()
        standard.close()
