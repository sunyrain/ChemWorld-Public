<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/getting_started.en.md; edit the upstream guide. -->

# Start with the programmable environment

[Environment API and user-owned rewards](environment.md) is the recommended entry.
The task examples below are optional research templates, with their own completion rules.
There is no required official agent score or leaderboard.

This page targets the current development runtime, not the frozen paper physics.
Software worlds are not instructions for physical experiments.
[中文](https://sunyrain.github.io/ChemWorld-Public/docs/zh/getting_started/) · [Distribution scope](community_release.md)

## Three entry points

1. **Use current features:** install a wheel built from main in an isolated environment:
   `python -m pip install /absolute/path/to/chemworld_bench-<version>-py3-none-any.whl`.
   A new PyPI release/tag has not been authorized. The old published 0.2.0 is not current main.
2. **Develop or run complete examples:** clone Git and use the committed lockfile below.
   Build from Git or a generated sdist, not a GitHub source ZIP.
3. **Reproduce paper results:** use the [independent frozen code and lockfile](https://github.com/sunyrain/ChemWorld-Public/tree/03e8026301c185fd6ba5bdbda7460765d9b3e724).
   Do not replay old trajectories under current physics.

Python 3.11/3.12 are validation targets; the distribution scope lists actual OS
results and unverified combinations. With uv already installed:

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
uv run --no-sync chemworld tasks list --format summary
```

## A final assay and your first custom agent

```bash
uv run --no-sync python examples/demo_manual_event_sequence.py
uv run --no-sync python examples/demo_agent_facing_api.py
uv run --no-sync python examples/demo_minimal_agent.py --output runs/minimal-agent.jsonl
uv run --no-sync chemworld verify --submission runs/minimal-agent.jsonl --tolerance 0
```

Choose a fresh output path: examples refuse to overwrite prior results.
`FourStepAgent` subclasses `BaseAgent` and implements `act(history)`. Replace that
method with your policy; this is a lifecycle example, not an optimizer or a
learned scientific model. The runner invokes reset/update and logs interactions.
It uses no provider key. `terminate` is not a final assay; a committed
`measure` with `instrument="final_assay"` closes the experiment. A low score is
still a result. Replay checks execution, not unrecorded reasoning.

These commands run in a checkout. With a wheel installed, use that environment's
`python` and `chemworld`. Example sources come with the repository
or sdist; a wheel does not create `examples/` in your current working directory.

## Complete offline research loop

```bash
uv run --no-sync python examples/demo_offline_research.py --output runs/offline-example
```

The [fixed example design](offline_research.md) includes seven episodes, notebook
writing, a host-owned fact ledger, forecasts sealed before held-out execution,
actual prediction errors, one expected rejection, an open batch, a truncated
batch, JSONL export and exact replay in seven new processes. It is a development
demonstration, not a paper rerun. Missing assays stay null, not zero; read
`observed_mask`. Optional Parquet backends are not validated by this JSONL example.

## Interface release deferred

Only Research Workbench and Blender Lab remain as internal interface prototypes.
They are excluded from current distributions. Use the Python API, environment CLI and contracts below.

## Public contract

```python
import gymnasium as gym
import chemworld

env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
try:
    _, info = env.reset(seed=0)
    print(info["task_id"])
    print(env.unwrapped.action_schema("heat"))
    print(env.unwrapped.available_actions())
finally:
    env.close()
```

`task_info()` declares support; `available_actions()` and `validate_action()` give
current legality. Use schema field names and units. See
[operations](operations.md) and the [contribution guide](https://github.com/sunyrain/ChemWorld/blob/main/CONTRIBUTING.md).
