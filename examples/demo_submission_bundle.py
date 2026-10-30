"""Build and validate a complete ChemWorld submission bundle example."""

from __future__ import annotations

import json
from pathlib import Path

from chemworld.legacy.submission_examples import (
    build_example_submission_bundle,
)


def main() -> None:
    result = build_example_submission_bundle(
        Path("runs") / "example_submission",
        task_id="reaction-to-purification",
        agent_name="tool_using_llm_stub",
        seeds=[0],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
