"""Generate, verify, and inspect one ChemWorld trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.legacy.agent_registry import make_agent


def main() -> None:
    output_path = Path("runs") / "demos" / "verify_random.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("random"),
        world_split="public-dev",
        budget=8,
        objective="balanced",
        seed=11,
        output_path=output_path,
    )

    records = load_jsonl(output_path)
    verification = verify_records(records)
    print(json.dumps(verification.to_dict(), indent=2, sort_keys=True))

    env = gym.make("ChemWorld", world_split="public-dev", budget=8, seed=11)
    try:
        env.reset(seed=11)
        constitution = env.unwrapped.constitution_summary()
        print(json.dumps({"constitution_rules": constitution["rules"]}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

