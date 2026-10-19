"""Run one explicit event-sequence experiment in ChemWorld."""

from __future__ import annotations

import json

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.data.logging import observation_to_json


def main() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", budget=8, seed=7)
    try:
        observation, task_info = env.reset(seed=7)
        print(json.dumps({"task": task_info["task_id"], "initial": _flat(observation)}))

        actions = [
            {"operation": "add_solvent", "volume_L": 0.030, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 388.0,
                "duration_s": 1500.0,
                "stirring_speed_rpm": 720.0,
            },
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "wait", "duration_s": 600.0, "stirring_speed_rpm": 720.0},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]

        final_assay = False
        for action in actions:
            observation, reward, terminated, truncated, info = env.step(action)
            committed = info["transaction_status"] == "committed"
            final_assay = bool(
                committed and terminated and action.get("instrument") == "final_assay"
            )
            print(
                json.dumps(
                    {
                        "step": info["step"],
                        "operation": info["operation_type"],
                        "reward": round(reward, 4),
                        "observation": _flat(observation),
                        "flags": info["constraint_flags"],
                        "transaction_status": info["transaction_status"],
                        "observed_mask": info["observed_mask"],
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                    sort_keys=True,
                )
            )
            if not committed or terminated or truncated:
                break
        print(json.dumps({"final_assay_completed": final_assay}))
        if not final_assay:
            raise RuntimeError(
                "Experiment did not reach a committed final assay; see step records."
            )
    finally:
        env.close()


def _flat(observation: dict[str, object]) -> dict[str, float | None]:
    return {
        key: None if value is None else round(value, 4)
        for key, value in observation_to_json(observation).items()
    }


if __name__ == "__main__":
    main()
