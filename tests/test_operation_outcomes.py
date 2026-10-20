"""Outcome propagation through actual transactions, sampling and resource settlement.

Stopped/process-failure injections below are contract fixtures around the existing
reactor; they do not introduce or qualify a new physical failure mechanism.
"""

from copy import deepcopy
from dataclasses import replace

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import make_agent, run_agent
from chemworld.eval.verify import verify_records
from chemworld.foundation import Observation, equipment_settings
from chemworld.physchem.batch_reactors import DynamicBatchReactorModel
from chemworld.runtime.kernel_contracts import ModelProviderResult


@pytest.fixture
def env():
    instance = gym.make(
        "ChemWorld", task_id="reaction-to-assay", seed=4, budget_override=50,
    ).unwrapped
    instance.reset(seed=4)
    for action in (
        {"operation": "add_solvent", "volume_L": 0.03, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.01},
        {"operation": "add_catalyst", "catalyst": 1, "catalyst_amount_mol": 0.0002},
    ):
        assert instance.step(action)[4]["operation_outcome"] == "success"
    yield instance
    instance.close()


def heat(env, duration=600.0):
    return env.step({
        "operation": "heat", "duration_s": duration, "target_temperature_K": 385.0,
        "stirring_speed_rpm": 700.0,
    })


def test_solver_failure_keeps_previous_commits_and_has_no_fake_temperature(env, monkeypatch):
    heat(env, 30.0)
    before = env._state

    def nonconvergence(*args, **kwargs):
        raise RuntimeError("controlled solver nonconvergence")

    monkeypatch.setattr(DynamicBatchReactorModel, "simulate", nonconvergence)
    info = heat(env)[4]
    after = env._state
    assert info["operation_outcome"] == "numerical_failure"
    assert info["rollback_reason"] == "numerical_failure"
    assert info["actual_duration_s"] == 0.0
    assert info["outcome_diagnostics"]["error_type"] == "RuntimeError"
    assert after.species_amounts == before.species_amounts
    assert after.temperature_K == before.temperature_K
    for key in ("time_s", "energy_jacket_J", "heat_reaction_J", "heat_loss_J"):
        assert getattr(after.ledger, key) == getattr(before.ledger, key)
    assert after.ledger.cost - before.ledger.cost == pytest.approx(0.03)
    assert all(check["passed"] for check in env._last_operation_record.constitution_checks)
    assert not any("temperature" in str(event) for event in info["world_events"])


@pytest.mark.parametrize("outcome", ["stopped", "process_failure"])
def test_valid_partial_outcomes_commit_real_state_and_only_actual_cost(env, monkeypatch, outcome):
    services = env.runtime.domain_services
    integrate = services.reaction_thermal.integrate

    def partial(state, action, **kwargs):
        actual = integrate(state, {**action, "duration_s": 5.0}, **kwargs)
        return replace(
            actual, outcome=outcome, success=outcome != "process_failure",
            failure_reason="contract_fixture_failure" if outcome == "process_failure" else None,
            diagnostics={"contract_fixture": True, "termination_reason": "fixture_event"},
        )

    monkeypatch.setattr(services.reaction_thermal, "integrate", partial)
    before = env._state
    _, _, terminated, _, info = heat(env)
    after = env._state
    assert info["transaction_status"] == "committed"
    assert info["operation_outcome"] == outcome
    assert info["actual_duration_s"] == 5.0
    assert after.ledger.time_s - before.ledger.time_s == 5.0
    assert after.ledger.cost - before.ledger.cost == pytest.approx(5.0 / 3600 * 0.03)
    assert after.species_amounts != before.species_amounts
    assert after.ledger.energy_jacket_J != before.ledger.energy_jacket_J
    assert after.temperature_K != before.temperature_K
    assert all(check["passed"] for check in env._last_operation_record.constitution_checks)
    assert not terminated  # The fixture does not define a task terminal condition.


def test_reactor_terminal_event_propagates_without_charging_requested_tail(env, monkeypatch):
    simulate = DynamicBatchReactorModel.simulate

    def event_stop(model, *args, **kwargs):
        result = simulate(model, *args, **{
            **kwargs, "duration_s": 7.0, "evaluation_times_s": (0.0, 7.0),
        })
        return replace(result, metadata={
            **result.metadata, "termination_reason": "contract_fixture_stop",
        })

    monkeypatch.setattr(DynamicBatchReactorModel, "simulate", event_stop)
    before = env._state
    info = heat(env)[4]
    assert info["operation_outcome"] == "stopped"
    assert info["actual_duration_s"] == 7.0
    assert env._state.ledger.cost - before.ledger.cost == pytest.approx(7 / 3600 * 0.03)


def test_candidate_clock_must_match_provider_duration_before_commit(env, monkeypatch):
    integrate = env.runtime.domain_services.reaction_thermal.integrate

    def incorrect_clock(*args, **kwargs):
        return replace(integrate(*args, **kwargs), actual_duration_s=1.0)

    monkeypatch.setattr(env.runtime.domain_services.reaction_thermal, "integrate", incorrect_clock)
    before = env._state
    info = heat(env)[4]
    assert info["operation_outcome"] == "numerical_failure"
    assert info["actual_duration_s"] == 0.0
    assert env._state.species_amounts == before.species_amounts
    assert env._state.ledger.time_s == before.ledger.time_s
    assert any(
        check["name"] == "operation_actual_duration" and not check["passed"]
        for check in env._last_operation_record.constitution_checks
    )


def test_rejected_request_has_only_one_existing_request_penalty(env):
    before = env._state
    info = env.step({"operation": "heat", "duration_s": -1.0})[4]
    assert info["operation_outcome"] == "rejected"
    assert info["actual_duration_s"] == 0.0
    assert env._state.ledger.cost - before.ledger.cost == pytest.approx(0.01)
    assert env._state.species_amounts == before.species_amounts


def test_low_yield_remains_a_valid_physical_success(env):
    info = heat(env, 1.0)[4]
    assert info["transaction_status"] == "committed"
    assert info["operation_outcome"] == "success"
    assert info["actual_duration_s"] == 1.0
    assert env.runtime.domain_services.species_view.truth_values(env._state)["yield"] < 0.1


@pytest.mark.parametrize("failure", ["domain", "packet"])
def test_invalid_readout_preserves_sampling_and_counts_instrument_use_once(monkeypatch, failure):
    card = CampaignResourceCard(
        card_id="invalid-readout", operation_attempt_limit=20, vessel_start_limit=1,
        final_assay_limit=1, nonfinal_instrument_use_limit=3, stock_limits={},
    )
    env = gym.make(
        "ChemWorld", task_id="reaction-to-assay", episode_mode_override="campaign",
        campaign_resource_card=card, budget_override=20,
    ).unwrapped
    try:
        env.reset(seed=3)
        env.step({"operation": "add_solvent", "volume_L": 0.03, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.01})
        env.step({"operation": "measure", "instrument": "hplc"})
        before = env._state
        cached = dict(before.process.last_observation)

        def invalid_readout(*args, **kwargs):
            if failure == "domain":
                raise ValueError("controlled readout outside instrument range")
            return Observation(
                values={"score": float("nan")}, units={"score": "dimensionless"},
                observed_mask={"score": True},
            )

        monkeypatch.setattr(env.observation_kernel, "observe", invalid_readout)
        _, reward, terminated, _, info = env.step({"operation": "measure", "instrument": "hplc"})
        after = env._state
        instrument = env.constitution.instruments["hplc"]
        assert info["operation_outcome"] == "success"
        assert info["transaction_status"] == "committed"
        assert info["measurement_status"] == "invalid"
        assert after.volume_L == pytest.approx(before.volume_L - instrument.sample_volume_L)
        assert after.ledger.sample_consumed_L - before.ledger.sample_consumed_L == pytest.approx(
            instrument.sample_volume_L,
        )
        assert after.ledger.cost - before.ledger.cost == pytest.approx(instrument.cost)
        assert info["measurement_cost"] == pytest.approx(instrument.cost)
        assert info["sample_consumed"] == pytest.approx(instrument.sample_volume_L)
        assert after.process.last_observation == cached
        assert info["campaign_resource_outcome_delta"]["nonfinal_instrument_uses"] == 1
        assert info["environment_reward"]["fresh_measurement"] is False
        assert info["raw_signal"] == {}
        assert reward == 0.0
        assert terminated is False
        settings = equipment_settings(after.equipment, "instrument:hplc")
        assert settings["use_count"] == 2
    finally:
        env.close()


def test_provider_errors_do_not_silently_become_physical_instrument_damage(env, monkeypatch):
    def programming_error(*args, **kwargs):
        raise KeyError("broken software contract")

    monkeypatch.setattr(env.observation_kernel, "observe", programming_error)
    with pytest.raises(KeyError, match="broken software contract"):
        env.step({"operation": "measure", "instrument": "hplc"})


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_provider_result_rejects_invalid_elapsed_time(duration):
    with pytest.raises(ValueError, match="actual_duration_s"):
        ModelProviderResult(outputs={}, diagnostics={}, actual_duration_s=duration)


def test_unexecuted_provider_result_cannot_report_consumed_physical_time():
    with pytest.raises(ValueError, match="unexecuted"):
        ModelProviderResult(
            outputs={}, diagnostics={}, outcome="numerical_failure", success=False,
            failure_reason="solver_failed", actual_duration_s=10.0,
        )


def test_kernel_exchange_and_consumption_are_the_committed_record(env):
    before = env._state
    result = env.runtime.apply_transaction(
        before, {"operation": "add_reagent", "amount_mol": 0.001},
    )
    kernel = result.kernel_result
    assert kernel.material_balance is result.operation_record.material_balance
    assert dict(kernel.material_balance.stock_consumption) == {"reagent_mol": 0.001}
    assert kernel.actual_consumption["cost"] == result.state.ledger.cost - before.ledger.cost
    assert kernel.actual_consumption["sample_consumed_L"] == 0.0
    assert kernel.actual_consumption["physical_time_s"] == 0.0


def test_outcome_and_duration_are_logged_and_checked_by_actual_replay(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    run_agent(
        env_id="ChemWorld", agent=make_agent("scripted_chemistry"),
        world_split="public-dev", budget=18, objective="balanced", seed=0,
        task_id="reaction-to-assay", output_path=path,
    )
    records = load_jsonl(path)
    assert verify_records(records, tolerance=0).verified
    assert all(row["operation_outcome"] in {"success", "rejected"} for row in records)
    assert any(row["actual_duration_s"] > 0 for row in records)
    for field, value in (
        ("operation_outcome", "process_failure"),
        ("actual_duration_s", 999.0),
        ("outcome_diagnostics", {"fabricated": True}),
        ("measurement_status", "invalid"),
    ):
        tampered = deepcopy(records)
        tampered[0][field] = value
        result = verify_records(tampered, tolerance=0)
        assert not result.verified
        assert any(str(item["field"]).startswith(field) for item in result.mismatches)
