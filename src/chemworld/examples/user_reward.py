"""A caller-owned policy and reward over the existing native runtime.

Run with ``python -m chemworld.examples.user_reward``. Replace ``choose_actions``
with your agent and ``pressure_reward`` with your own objective, or omit rewards.
No benchmark registry, baseline agent, model provider or leaderboard is used.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec


def choose_actions() -> Iterable[dict[str, Any]]:
    """A small demonstration policy, not an optimization method."""
    yield {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5}
    yield {"operation": "add_component", "component": "O2-feed", "amount_mol": 3e-5}
    yield {"operation": "measure", "instrument": "gas_pressure"}


def pressure_reward(action: dict[str, Any], observation: dict[str, Any] | None) -> float | None:
    """Example user objective: proximity to 120 kPa using an acquired reading.

    The target and 1000 Pa scale are arbitrary user choices, not ChemWorld scores.
    No new measurement means no reward observation, rather than a fabricated zero.
    """
    del action
    if observation is None:
        return None
    for channel in observation["native"]["channels"]:
        if channel["channel_id"] == "pressure" and channel["quality"] == "valid":
            return -abs(float(channel["value"]) - 120_000.0) / 1000.0
    return None


def run_example(
    actions: Iterable[dict[str, Any]] | None = None,
    reward_fn: Callable[[dict[str, Any], dict[str, Any] | None], float | None]
    | None = pressure_reward,
) -> dict[str, Any]:
    session = ChemWorldRuntime(composition=surface_world_spec())
    rows = []
    try:
        session.reset(seed=12)
        for action in choose_actions() if actions is None else actions:
            result = session.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(f"Example action rejected: {action}")
            # A cached observation is not a fresh measurement. Only publish this
            # operation's valid measurement to the optional user reward function.
            observation = (
                result.observation.to_dict()
                if action["operation"] == "measure" and result.info["measurement_status"] == "valid"
                else None
            )
            rows.append(
                {
                    "action": action,
                    "observation": observation,
                    "user_reward": reward_fn(action, observation) if reward_fn else None,
                }
            )
            if result.terminated or result.truncated:
                break
    finally:
        session.close()
    return {"reward_owner": "caller", "steps": rows}


def main() -> None:
    print(json.dumps(run_example(), indent=2))


if __name__ == "__main__":
    main()
