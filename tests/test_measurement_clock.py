from __future__ import annotations

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.eval.verify import verify_records
from chemworld.world.instruments import instrument_contracts


def measure_path(instrument, path):
    task = "equilibrium-characterization" if instrument == "ph_meter" else "reaction-to-assay"
    kwargs = {}
    if instrument == "particle_size":
        task = "reaction-to-crystallization"
        kwargs["full_process_contract_id"] = "phase-resolved-process-v5"
    env = gym.make(
        "ChemWorld",
        task_id=task,
        seed=0,
        budget_override=40,
        episode_mode_override="single_experiment",
        **kwargs,
    )
    try:
        env.reset(seed=0)
        actions = [
            {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.012},
        ]
        if instrument == "particle_size":
            actions.extend(
                [
                    {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": 0},
                    {
                        "operation": "heat",
                        "target_temperature_K": 380.0,
                        "duration_s": 1800.0,
                        "stirring_speed_rpm": 600.0,
                    },
                    {"operation": "quench"},
                    {
                        "operation": "cool_crystallize",
                        "target_temperature_K": 320.0,
                        "duration_s": 3600.0,
                    },
                ]
            )
        if instrument == "final_assay":
            actions.append({"operation": "terminate"})
        actions.append({"operation": "measure", "instrument": instrument})
        with TrajectoryLogger(path) as logger:
            task_info = {**env.unwrapped.task_info(), **env.unwrapped.evaluator_provenance()}
            for step, action in enumerate(actions, 1):
                before = env.unwrapped._state
                observation, reward, terminated, truncated, info = env.step(action)
                assert info["transaction_status"] == "committed", info.get("preconditions")
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
        after = env.unwrapped._state
        contract = instrument_contracts(include_particle_size=True)[instrument].to_dict()
        assert contract["latency_s"] == 0.0
        acquisition = info["raw_signal"]["acquisition"]
        assert acquisition == info["outcome_diagnostics"]["acquisition"]
        assert acquisition["clock_semantics"] == contract["clock_semantics"]
        assert acquisition["sample_time_s"] == before.ledger.time_s
        assert acquisition["result_time_s"] == after.ledger.time_s
        assert after.ledger.time_s == before.ledger.time_s
        assert after.temperature_K == before.temperature_K
        for field in ("energy_jacket_J", "heat_reaction_J", "heat_loss_J"):
            assert getattr(before.ledger, field) == getattr(after.ledger, field)
        assert after.ledger.cost - before.ledger.cost == pytest.approx(contract["cost"], abs=1e-12)
        assert before.volume_L - after.volume_L == pytest.approx(
            contract["sample_consumption_L"], abs=1e-12
        )
        assert after.ledger.sample_consumed_L - before.ledger.sample_consumed_L == pytest.approx(
            contract["sample_consumption_L"],
            abs=1e-12,
        )
        records = load_jsonl(path)
        assert verify_records(records, tolerance=0.0).verified
        return {
            "instrument": instrument,
            "clock": info["raw_signal"]["acquisition"],
            "sample_consumed_L": info["sample_consumed"],
            "cost": info["measurement_cost"],
            "steps": len(records),
            "exact_replay": True,
        }
    finally:
        env.close()


@pytest.mark.parametrize(
    "instrument", ["hplc", "gc", "uvvis", "ph_meter", "final_assay", "particle_size"]
)
def test_measurement_clock_matches_sampling_return_and_resources(instrument, tmp_path):
    measure_path(instrument, tmp_path / f"{instrument}.jsonl")


def test_rejected_measurement_has_no_acquisition_or_instrument_consumption():
    env = gym.make("ChemWorld", task_id="reaction-to-assay")
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "hplc"})
        assert info["transaction_status"] == "rolled_back"
        assert info["raw_signal"] == {}
        assert info["measurement_cost"] == 0.0
        assert info["sample_consumed"] == 0.0
        assert env.unwrapped._state.ledger.time_s == 0.0
    finally:
        env.close()
