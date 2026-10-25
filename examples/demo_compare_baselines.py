"""Run a tiny local benchmark suite and aggregate leaderboard rows."""

from __future__ import annotations

import json
from pathlib import Path

from chemworld.eval.suite import run_suite
from chemworld.legacy.leaderboard import aggregate_leaderboard


def main() -> None:
    output_root = Path("runs") / "demos" / "baseline_compare"
    agents = ["random", "scripted_chemistry", "lhs"]
    results: list[dict[str, object]] = []

    for agent in agents:
        results.extend(
            run_suite(
                agent_name=agent,
                env_id="ChemWorld",
                world_splits=["public-test", "private-eval"],
                seeds=[0, 1],
                budget=12,
                objective="balanced",
                output_dir=output_root / agent,
            )
        )

    print(json.dumps(aggregate_leaderboard(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

