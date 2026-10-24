"""Measurements consume the same physical history as heat/wait and preserve specimens."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.foundation import Observation
from chemworld.foundation.observations import ObservationChannel
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.surface_world import surface_world_spec


def env_for(spec=None, **kwargs):
    env = ChemWorldEnv(composition=spec or surface_world_spec(**kwargs))
    env.reset(seed=12)
    for name, amount in (("CO-feed", 8e-5), ("O2-feed", 4e-5)):
        perform(env, operation="add_component", component=name, amount_mol=amount)
    return env


def perform(env, **action):
    result = env.step(action)
    assert result[4]["transaction_status"] == "committed", result[4]
    return result


def sensor_variant(spec, name, *, observer=None, **timing):
    modules = DomainModuleRegistry(
        tuple(
            replace(
                module,
                instruments=tuple(
                    replace(
                        instrument,
                        contract=replace(instrument.contract, **timing),
                        native_observer=observer or instrument.native_observer,
                    )
                    if instrument.definition.id == name
                    else instrument
                    for instrument in module.instruments
                ),
            )
            for module in spec.modules.modules
        )
    )
    return replace(spec, modules=modules)


@pytest.mark.parametrize("name,seconds", [("gas_pressure", 1.0), ("gas_composition", 5.0)])
def test_in_situ_acquisition_is_the_same_physical_advance_as_wait(name, seconds):
    env, reference = env_for(), env_for()
    before = env._state
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(env._rng.bit_generator.state)
    perform(reference, operation="wait", duration_s=seconds, stirring_speed_rpm=600)
    result = perform(env, operation="measure", instrument=name)
    assert result[4]["measurement_status"] == "valid"
    assert env._state.species_amounts == reference._state.species_amounts
    assert env._state.state_slots == reference._state.state_slots
    assert env._state.temperature_K == reference._state.temperature_K
    assert env._state.ledger.energy_jacket_J == reference._state.ledger.energy_jacket_J
    assert env._state.species_amounts["CO2"] > before.species_amounts["CO2"]
    declaration = env.runtime.modules.instrument_specs()[name]
    expected = declaration.native_observer(reference._state, {}, rng)
    record = env._state.process.last_observation_record
    assert [c.value for c in record.channels] == [c.value for c in expected.channels]
    assert env._state.ledger.cost - before.ledger.cost == pytest.approx(declaration.definition.cost)
    assert env._state.ledger.time_s == seconds
    assert env._state.ledger.analysis_time_s == 0
    assert all(
        c.sampled_at.time_s == seconds and c.available_at == c.sampled_at for c in record.channels
    )
    assert all(c.sampled_at.clock_id == "vessel:reactor" for c in record.channels)
    assert env._state.vessel_elapsed_s["reactor"] == seconds
    assert env._state.vessel_elapsed_s["receiver"] == 0


@pytest.mark.parametrize("host_evolves", [False, True])
def test_offline_analysis_uses_sample_snapshot_and_explicit_host_policy(host_evolves):
    spec = sensor_variant(
        surface_world_spec(),
        "gas_composition",
        clock_semantics="offline_snapshot",
        latency_s=0,
        analysis_time_s=5,
        host_evolves_during_analysis=host_evolves,
    )
    env = env_for(spec)
    before = env._state
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(env._rng.bit_generator.state)
    expected = spec.modules.instrument_specs()["gas_composition"].native_observer(before, {}, rng)
    info = perform(env, operation="measure", instrument="gas_composition")[4]
    record = env._state.process.last_observation_record
    assert [c.value for c in record.channels] == [c.value for c in expected.channels]
    assert env._state.ledger.time_s == (5 if host_evolves else 0)
    assert env._state.ledger.analysis_time_s == 5
    assert env._state.ledger.cost - before.ledger.cost == pytest.approx(0.08)
    assert (env._state.species_amounts != before.species_amounts) == host_evolves
    for channel in record.channels:
        assert channel.sampled_at.time_s == 0
        assert channel.available_at.time_s == 5
        assert channel.available_at.clock_id == (
            "vessel:reactor" if host_evolves else "reactor:instrument:gas_composition"
        )
    assert info["outcome_diagnostics"]["acquisition"]["analysis_effort_s"] == 5


def test_terminal_assay_reads_frozen_specimen_and_preserves_devices():
    env = env_for(terminal_assay=True)
    perform(env, operation="measure", instrument="gas_composition")
    assert not perform(env, operation="terminate")[2]
    frozen = env._state
    _, _, done, truncated, info = perform(env, operation="measure", instrument="gas_final_assay")
    assert done and not truncated
    assert env._last_operation_record.is_final_assay
    assert info["measurement_status"] == "valid"
    assert info["terminal_assay"] is True
    assert info["leaderboard_score"] is None  # This task requests observations, not a yield score.
    assert env._state.species_amounts == frozen.species_amounts
    assert env._state.state_slots == frozen.state_slots
    assert env._state.temperature_K == frozen.temperature_K
    assert env._state.ledger.time_s == frozen.ledger.time_s
    assert env._state.ledger.analysis_time_s - frozen.ledger.analysis_time_s == 5
    for key, record in frozen.equipment.equipment.items():
        assert env._state.equipment.equipment[key] == record
    repeated = env.runtime.apply_transaction(
        env._state, {"operation": "measure", "instrument": "gas_final_assay"}
    )
    assert repeated.kernel_result.outcome == "rejected"
    assert repeated.state.ledger.analysis_time_s == env._state.ledger.analysis_time_s


def test_assay_is_not_required_for_explicit_end_and_cannot_run_before_freeze():
    env = env_for()
    devices = env._state.equipment
    assert perform(env, operation="terminate")[2]
    assert env._state.equipment == devices
    env = env_for(terminal_assay=True)
    info = env.step({"operation": "measure", "instrument": "gas_final_assay"})[4]
    assert info["operation_outcome"] == "rejected"
    assert env._state.ledger.time_s == env._state.ledger.analysis_time_s == 0


def bad_readout(state, action, rng):
    raise ValueError("controlled readout failure")


def test_readout_failure_keeps_actual_reaction_cost_and_instrument_use():
    env = env_for(sensor_variant(surface_world_spec(), "gas_composition", observer=bad_readout))
    before = env._state
    info = perform(env, operation="measure", instrument="gas_composition")[4]
    assert info["measurement_status"] == "invalid"
    assert env._state.ledger.time_s == 5
    assert env._state.ledger.cost - before.ledger.cost == pytest.approx(0.08)
    assert env._state.species_amounts["CO2"] > 0
    assert env._state.equipment.equipment["instrument:gas_composition"].settings["use_count"] == 1
    assert env._state.process.last_observation_record == before.process.last_observation_record


def saturated_readout(state, action, rng):
    return Observation(
        channels=(
            ObservationChannel(
                "pressure",
                "Pa",
                100000.0,
                quality="saturated",
                valid_range=(0, 100000.0),
            ),
        ),
        instrument_id="gas_pressure",
        cost=0.02,
    )


def test_saturated_packet_is_flagged_without_reversing_physical_experience():
    env = env_for(sensor_variant(surface_world_spec(), "gas_pressure", observer=saturated_readout))
    perform(env, operation="measure", instrument="gas_pressure")
    assert env._state.ledger.time_s == 1
    assert env._state.process.last_observation_record.channel("pressure").quality == "saturated"


def test_numerical_failure_rolls_back_acquisition_and_does_not_use_instrument(monkeypatch):
    from chemworld.physchem.coupled_system import CoupledSystem

    env = env_for()
    before = env._state

    def fail(*args, **kwargs):
        raise RuntimeError("controlled solver failure")

    monkeypatch.setattr(CoupledSystem, "advance", fail)
    info = env.step({"operation": "measure", "instrument": "gas_composition"})[4]
    assert info["operation_outcome"] == "numerical_failure"
    assert info["actual_duration_s"] == 0
    assert info["outcome_diagnostics"]["solver_error"] == "controlled solver failure"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.equipment == before.equipment
    assert env._state.ledger.time_s == env._state.ledger.analysis_time_s == 0


def test_physical_stop_commits_actual_endpoint_without_inventing_complete_readout():
    spec = surface_world_spec(include_surface=False)
    models = dict(spec.physics.models_of('thermal'))
    models["reactor"] = replace(models["reactor"], minimum_temperature_K=349.99)
    components = tuple(
        replace(component, parameters={**component.parameters,
                "temperature_range_K": [349.99, 430]})
        if component.kind == "thermal" and component.vessel == "reactor" else component
        for component in spec.components
    )
    env = env_for(
        replace(
            spec,
            components=components,
            physics=replace(spec.physics, models={**spec.physics.models, "thermal": models}),
        )
    )
    perform(env, operation="pause_control")
    info = perform(env, operation="measure", instrument="gas_composition")[4]
    assert info["operation_outcome"] == "stopped"
    assert info["measurement_status"] == "invalid"
    assert 0 < env._state.ledger.time_s < 5
    assert env._state.temperature_K == pytest.approx(349.99)
    assert not info["outcome_diagnostics"]["acquisition"]["acquisition_complete"]


def test_cached_readout_retains_sample_time_without_new_noise_or_cost_and_mix_invalidates():
    env = env_for()
    perform(env, operation="measure", instrument="gas_pressure")
    record = env._state.process.last_observation_record
    before, rng = env._state, deepcopy(env._rng.bit_generator.state)
    for _ in range(3):
        cached = env.observation_kernel.observe(env._state, {"operation": "wait"}, env._rng)
        assert cached.channels == record.channels
    assert env._state == before and env._rng.bit_generator.state == rng
    perform(env, operation="wait", duration_s=2, stirring_speed_rpm=600)
    assert env._state.process.last_observation_record.channel("pressure").sampled_at.time_s == 1
    perform(
        env,
        operation="route_material",
        connection="reactor-to-receiver",
        transfer_fraction=0.5,
        mixing="allow",
    )
    assert env._state.process.last_observation_record is None or not any(
        c.channel_id == "pressure" for c in env._state.process.last_observation_record.channels
    )


def test_local_sampling_clock_does_not_include_other_vessels_elapsed_time():
    env = env_for()
    perform(env, operation="measure", instrument="gas_composition")
    perform(env, operation="select_vessel", vessel="receiver")
    perform(env, operation="measure", instrument="gas_pressure")
    channel = env._state.process.last_observation_record.channel("pressure")
    assert env._state.ledger.time_s == 6
    assert channel.sampled_at.clock_id == "vessel:receiver" and channel.sampled_at.time_s == 1
    assert channel.sample_id.startswith("receiver:")


def test_timed_instrument_requires_physical_clock_at_world_binding():
    from chemworld.world.composition import WorldCompositionError, compile_world_composition

    spec = surface_world_spec()
    # A timed instrument cannot bind when its physical system has no solver;
    # the shared owner check now diagnoses this before task capability checks.
    with pytest.raises(WorldCompositionError, match="no declared physical advance strategy"):
        compile_world_composition(
            replace(
                spec, physics=replace(spec.physics, models={**spec.physics.models, "thermal": {}})
            )
        )


@pytest.mark.parametrize("noise_mode", ["sequential", "keyed"])
def test_measured_history_replays_native_timestamps_noise_and_terminal_role(tmp_path, noise_mode):
    from chemworld.agent_interface import agent_view_bundle
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.eval.verify import verify_records

    env = ChemWorldEnv(
        composition=surface_world_spec(terminal_assay=True), observation_noise_mode=noise_mode
    )
    env.reset(seed=12)
    task_info = {**env.task_info(), **env.evaluator_provenance()}
    path = tmp_path / "measurements.jsonl"
    actions = [
        {"operation": "add_component", "component": "CO-feed", "amount_mol": 8e-5},
        {"operation": "add_component", "component": "O2-feed", "amount_mol": 4e-5},
        {"operation": "measure", "instrument": "gas_pressure"},
        {"operation": "measure", "instrument": "gas_composition"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "gas_final_assay"},
    ]
    with TrajectoryLogger(path) as logger:
        for index, action in enumerate(actions, 1):
            obs, reward, done, truncated, info = perform(env, **action)
            logger.log(
                task_info=task_info,
                step=index,
                action=action,
                observation=observation_to_json(obs),
                reward=reward,
                terminated=done,
                truncated=truncated,
                info=info,
                agent_metadata={},
                agent_view=agent_view_bundle(env, obs, info),
            )
    report = verify_records(load_jsonl(path))
    assert report.verified, report
    records = load_jsonl(path)
    assert records[-1]["terminal_assay"] is True
    records[-1]["terminal_assay"] = False
    assert not verify_records(records).verified


def test_measurement_latency_uses_declared_time_budget_once():
    from chemworld.world.process_time_budget import ProcessTimeBudgetPolicy

    policy = ProcessTimeBudgetPolicy(
        "measurement-budget",
        "native",
        6,
        0,
        6,
        0,
        {},
        {},
        {},
        {},
        {},
        6,
    )
    spec = surface_world_spec()
    spec = replace(
        spec,
        task=replace(
            spec.task,
            resources={
                "time_s": 6,
                "process_time_policy": policy.to_dict(),
            },
        ),
    )
    env = env_for(spec)
    perform(env, operation="measure", instrument="gas_composition")
    perform(env, operation="measure", instrument="gas_pressure")
    before = env._state
    info = env.step({"operation": "measure", "instrument": "gas_pressure"})[4]
    assert info["operation_outcome"] == "rejected"
    assert info["declared_process_time_preflight"]["proposed_delta_s"] == 1
    assert env._state.ledger.time_s == 6
    assert env._state.equipment == before.equipment


@pytest.mark.parametrize(
    "overrides",
    [
        {"clock_semantics": "unknown"},
        {"latency_s": float("nan")},
        {"clock_semantics": "instantaneous_snapshot"},
        {"analysis_time_s": 1.0},
        {"requires_terminated": True},
        {"sample_consumption_L": 0.001},
    ],
)
def test_measurement_plan_rejects_unsupported_or_inconsistent_timing(overrides):
    contract = surface_world_spec().modules.instrument_specs()["gas_pressure"].contract
    with pytest.raises(ValueError):
        replace(contract, **overrides)


def test_material_invalidation_cannot_be_undone_by_environment_cache_refill():
    from chemworld.foundation.state import ProcessLedger

    env = env_for()
    # A compatibility-cache fixture reproduces the collaborator's finding;
    # the following feed and wait both execute the real shared runtime.
    env._state = env._state.replace(
        process=ProcessLedger(
            last_observation={"yield": 0.3},
            last_observed_mask={"yield": True},
        )
    )
    perform(env, operation="add_component", component="CO-feed", amount_mol=1e-6)
    assert env._state.process.last_observation_record is None
    observation, _, _, _, info = perform(
        env, operation="wait", duration_s=1, stirring_speed_rpm=600
    )
    assert not info["observed_mask"]["yield"]
    assert np.isnan(observation["yield"][0])
    assert env._state.process.last_observation_record is None


@pytest.mark.parametrize("terminal", [False, True])
def test_agent_lifecycle_instructions_follow_bound_task(terminal):
    from chemworld.agent_interface import task_prompt

    env = env_for(terminal_assay=terminal)
    lifecycle = task_prompt(env)["experiment_lifecycle"]
    if terminal:
        assert "gas_final_assay" in lifecycle["final_assay_action_template"]
        assert "frozen" in lifecycle["final_assay_effect"]
    else:
        assert "completes" in lifecycle["terminate_effect"]
        assert "No terminal assay" in lifecycle["final_assay_action_template"]
