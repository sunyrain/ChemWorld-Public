from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.verify import verify_records
from chemworld.foundation.state import PhaseLedger
from chemworld.physchem.reactor_shared import HeatTransferSpec
from chemworld.physchem.reactor_solvers import _jacket_heat_w
from chemworld.runtime.full_process_contract import unrecoverable_crystallization_result
from chemworld.world.thermal_control import REACTION_THERMAL_CONTROL_ID

CHARGE = [
    {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
    {"operation": "add_reagent", "amount_mol": 0.012},
    {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": 0},
]


def _heat(temperature, duration):
    return {
        "operation": "heat",
        "target_temperature_K": temperature,
        "duration_s": duration,
        "stirring_speed_rpm": 600.0,
    }


def _commit(env, action):
    result = env.step(action)
    assert result[-1]["transaction_status"] == "committed", result[-1].get("preconditions")
    return result[-1]


@pytest.mark.parametrize(
    "temperature,expected",
    [(250.0, 90.0), (295.0, 20.0), (300.0, 0.0), (305.0, -20.0), (400.0, -70.0)],
)
def test_continuous_jacket_limits_are_shared_by_static_and_programmed_solvers(
    temperature,
    expected,
):
    thermal = HeatTransferSpec(
        jacket_temperature_K=300.0,
        jacket_ua_W_per_K=4.0,
        maximum_jacket_heating_W=90.0,
        maximum_jacket_cooling_W=70.0,
    )
    assert thermal.jacket_heat_w(temperature) == expected
    assert (
        _jacket_heat_w(thermal, temperature_K=temperature, jacket_temperature_K=300.0) == expected
    )
    assert _jacket_heat_w(thermal, temperature_K=temperature, jacket_temperature_K=None) == 0.0
    assert thermal.jacket_heat_w(299.999) == pytest.approx(0.004)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("profile", ["heat", "cool", "initial_setpoint"])
def test_reaction_segmentation_preserves_material_temperature_and_energy(seed, profile):
    states = []
    for splits in (1, 6):
        env = gym.make("ChemWorld", task_id="reaction-to-assay", budget_override=40, seed=seed)
        try:
            env.reset(seed=seed)
            for action in CHARGE:
                _commit(env, action)
            initial_temperature = env.unwrapped._state.temperature_K
            if profile == "cool":
                _commit(env, _heat(385.0, 1200.0))
            target = {"heat": 365.0, "cool": 280.0, "initial_setpoint": initial_temperature}[
                profile
            ]
            for _ in range(splits):
                _commit(env, _heat(target, 1200.0 / splits))
            states.append(env.unwrapped._state)
        finally:
            env.close()
    left, right = states
    assert left.temperature_K == pytest.approx(right.temperature_K, abs=0.002, rel=0.0)
    assert left.species_amounts == pytest.approx(right.species_amounts, abs=2e-7, rel=0.0)
    assert left.ledger.time_s == pytest.approx(right.ledger.time_s, abs=1e-8, rel=0.0)
    assert left.ledger.cost == pytest.approx(right.ledger.cost, abs=1e-10, rel=0.0)
    for name in ("energy_jacket_J", "heat_reaction_J", "heat_loss_J"):
        assert getattr(left.ledger, name) == pytest.approx(
            getattr(right.ledger, name),
            abs=0.05,
            rel=2e-5,
        )


def _composition():
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "research-thermal-replay",
        "world_split": "public-dev",
        "components": [
            {"kind": kind, "role": role, "parameters": {}}
            for kind, role in (
                ("reaction", "transformation"),
                ("thermal", "temperature-and-energy"),
                ("observation", "public-measurement"),
            )
        ],
        "task": {
            "budget": 20,
            "resources": {"operation_budget": 20},
            "description": "Continuous thermal control replay.",
        },
    }


def _logged_actions(env, path, actions):
    with TrajectoryLogger(path) as logger:
        task_info = {**env.unwrapped.task_info(), **env.unwrapped.evaluator_provenance()}
        for step, action in enumerate(actions, 1):
            observation, reward, terminated, truncated, info = env.step(action)
            logger.log(
                task_info=task_info,
                step=step,
                action=action,
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={},
            )
    return load_jsonl(path)


@pytest.mark.parametrize("flavour", ["task", "taskless", "composition"])
def test_current_physics_replays_and_historical_physics_points_to_frozen_runtime(tmp_path, flavour):
    kwargs = (
        {"task_id": "reaction-to-assay"}
        if flavour == "task"
        else ({"composition": _composition()} if flavour == "composition" else {})
    )
    env = gym.make("ChemWorld", **kwargs)
    try:
        env.reset(seed=0)
        records = _logged_actions(
            env, tmp_path / "trajectory.jsonl", [*CHARGE, _heat(365.0, 1200.0)]
        )
        assert all(r["reaction_thermal_control_id"] == REACTION_THERMAL_CONTROL_ID for r in records)
        assert verify_records(records, tolerance=0.0).verified
        for historical_id in (None, "action-start-ua-v1", "unknown"):
            changed = deepcopy(records)
            if historical_id is None:
                for record in changed:
                    record.pop("reaction_thermal_control_id")
            else:
                changed[-1]["reaction_thermal_control_id"] = historical_id
            result = verify_records(changed, tolerance=0.0)
            assert not result.verified and result.checked_steps == 0
            assert "frozen runtime" in result.mismatches[0]["reason"]
    finally:
        env.close()


def _crystallization_actions(*, seeded=False, target=278.15):
    return (
        CHARGE
        + [_heat(380.0, 1800.0), {"operation": "quench"}]
        + ([{"operation": "seed_crystals", "seed_mass_g": 0.001}] if seeded else [])
        + [{"operation": "cool_crystallize", "target_temperature_K": target, "duration_s": 3600.0}]
    )


def _crystallization_env(seed=0):
    return gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=seed,
        full_process_contract_id="phase-resolved-process-v5",
        budget_override=40,
        episode_mode_override="single_experiment",
    )


def test_negative_result_closes_with_missing_statistics_and_replay(tmp_path):
    env = _crystallization_env()
    try:
        env.reset(seed=0)
        records = _logged_actions(
            env,
            tmp_path / "negative.jsonl",
            [
                *_crystallization_actions(target=320.0),
                {"operation": "measure", "instrument": "particle_size"},
                {"operation": "filter_crystals"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ],
        )
        particle, failed_filter, terminate, assay = records[-4:]
        assert particle["transaction_status"] == "committed"
        assert particle["raw_signal"]["d50_um"] is None
        assert particle["measurement_cost"] > 0.0
        assert particle["sample_consumed"] == 0.0
        assert particle["state_delta_summary"]["delta_time_s"] == 0.0
        assert failed_filter["transaction_status"] == "rolled_back"
        assert failed_filter["preconditions"]["filter_requires_crystallization"] is False
        assert terminate["transaction_status"] == "committed"
        assert assay["transaction_status"] == "committed" and assay["terminated"]
        assert assay["raw_signal"]["sample_outcome"]["reason"] == "below_crystal_recovery_threshold"
        for key in (
            "crystal_size",
            "crystal_purity",
            "crystal_csd_quality",
            "crystal_fines_fraction",
        ):
            assert (
                trajectory_value(assay, "observation")[key] is None
                and not assay["observed_mask"][key]
            )
            assert key + "_std" not in assay["uncertainty"]
        assert assay["observed_mask"]["crystal_yield"]
        assert sum(env.unwrapped._state.phases.phases["solid"].species_amounts_mol.values()) == 0.0
        assert assay["measurement_cost"] > 0.0 and assay["sample_consumed"] > 0.0
        assert env.unwrapped._state.ledger.cost > assay["measurement_cost"]
        assert verify_records(records, tolerance=0.0).verified
        changed = deepcopy(records)
        changed[-1]["raw_signal"]["sample_outcome"]["reason"] = "invented_positive_result"
        assert not verify_records(changed, tolerance=0.0).verified
        assert not env.unwrapped.validate_action(
            {"operation": "measure", "instrument": "final_assay"}
        )["valid"]
    finally:
        env.close()


def test_current_threshold_defines_negative_results_and_does_not_allow_early_closure():
    env = _crystallization_env()
    try:
        env.reset(seed=0)
        for action in _crystallization_actions():
            if action["operation"] == "cool_crystallize":
                assert not env.unwrapped.validate_action({"operation": "terminate"})["valid"]
            _commit(env, action)
        state = env.unwrapped._state
        tolerance = env.unwrapped.constitution.tolerance
        solid = state.phases.phases["solid"]
        for amount, expected in ((0.0, True), (tolerance, True), (tolerance * 1.001, False)):
            phases = {
                **state.phases.phases,
                "solid": replace(solid, species_amounts_mol={"P": amount}),
            }
            candidate = state.replace(phases=PhaseLedger(phases))
            assert unrecoverable_crystallization_result(candidate, tolerance=tolerance) is expected
        env.reset(seed=0)
        assert not env.unwrapped.validate_action({"operation": "terminate"})["valid"]
    finally:
        env.close()


def test_positive_result_still_requires_filtration():
    env = _crystallization_env(seed=1)
    try:
        env.reset(seed=1)
        for action in _crystallization_actions(seeded=True):
            _commit(env, action)
        assert not env.unwrapped.validate_action({"operation": "terminate"})["valid"]
        _commit(env, {"operation": "measure", "instrument": "particle_size"})
        _commit(env, {"operation": "filter_crystals"})
        _commit(env, {"operation": "terminate"})
        info = _commit(env, {"operation": "measure", "instrument": "final_assay"})
        assert "sample_outcome" not in info["raw_signal"]
        assert info["observed_mask"]["crystal_size"]
    finally:
        env.close()


def test_campaign_summary_retains_negative_result_after_next_batch_reset():
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        full_process_contract_id="phase-resolved-process-v5",
        budget_override=40,
        episode_mode_override="campaign",
    )
    try:
        env.reset(seed=0)
        for action in _crystallization_actions(target=320.0):
            _commit(env, action)
        _commit(env, {"operation": "terminate"})
        info = _commit(env, {"operation": "measure", "instrument": "final_assay"})
        summary = info["last_terminal_summary"]
        assert summary["final_assay"]
        assert summary["sample_outcome"]["status"] == "negative_result"
        assert summary["cost"] > 0.0
        assert not env.unwrapped._state.terminated
        assert env.unwrapped.campaign_state()["experiment_summaries"][-1] == summary
    finally:
        env.close()
